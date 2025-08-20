import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import numpy as np
import seaborn as sns
from PIL import Image
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.baseda import *
from utils.clip_part import *
from utils.CCDA import *
from trainers.imagenet_templates import  IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.SRDA.N_CTX
        ctx_init = cfg.TRAINER.SRDA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.name = cfg.DATASET.NAME
        self.tp = cfg.TRAINER.SRDA.TP
        self.vp = cfg.TRAINER.SRDA.VP
        self.t_deep = cfg.TRAINER.SRDA.T_DEEP
        self.v_deep = cfg.TRAINER.SRDA.V_DEEP
        self.deep_share = cfg.TRAINER.SRDA.DEEP_SHARED
        self.share_layer = cfg.TRAINER.SRDA.SHARE_LAYER
        self.num_tokens = cfg.TRAINER.SRDA.NUM_TOKENS    # number of prompted tokens
        self.deep_layer = cfg.TRAINER.SRDA.DEEP_LAYERS # num of layer has prompt ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.SRDA.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.SRDA.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = cfg.MODEL.HIDDEN_SIZE    # visual encoder hiden size(768)
        # 动态特征库参数
        self.Q = cfg.TRAINER.SRDA.Q
        self.w_s = cfg.TRAINER.SRDA.W_S
        self.w_low = cfg.TRAINER.SRDA.W_LOW
        self.w_high = cfg.TRAINER.SRDA.W_HIGH
        self.alpha = cfg.TRAINER.SRDA.ALPHA
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        # 初始化特征库
        self.source_feat_bank = [torch.zeros((self.Q, self.dim), device='cuda',dtype=dtype) for _ in range(n_cls)]
        self.target_feat_bank = [torch.zeros((self.Q, self.dim), device='cuda',dtype=dtype) for _ in range(n_cls)]
        self.source_max_probs = [torch.zeros(self.Q, device='cuda',dtype=dtype) for _ in range(n_cls)]
        self.target_max_probs = [torch.zeros(self.Q, device='cuda',dtype=dtype) for _ in range(n_cls)]

        # 初始化特征中心
        self.source_center = torch.zeros(n_cls, self.dim, device='cuda',dtype=dtype)
        self.target_center = torch.zeros(n_cls, self.dim, device='cuda',dtype=dtype)
        self.source_bank_counts=torch.zeros(n_cls,dtype=torch.long,device='cuda')
        self.target_bank_counts = torch.zeros(n_cls, dtype=torch.long, device='cuda')
        self.bank_updates=0
        self.bank_updates_time=0

        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:   # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ") 
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)
                
        self.vctx = None
        self.proj = None
        if self.vp:
            if self.share_layer != None:
                if self.share_layer[0] == 0:
                    self.proj = nn.Linear(ctx_dim, self.hidden_size).half()
                else:
                    vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                    nn.init.normal_(vctx_vectors, std=0.02)
                    self.vctx = nn.Parameter(vctx_vectors)
            else:
                vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                nn.init.normal_(vctx_vectors, std=0.02)
                self.vctx = nn.Parameter(vctx_vectors)
        
        self.deep_ctx = None
        if self.t_deep:
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, ctx_dim)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)
        
        self.deep_vctx = None
        if self.v_deep and not self.deep_share:    
            if self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
            elif self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] - 1, self.num_tokens, self.hidden_size)
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors) 
            
        elif self.v_deep and self.deep_share:
            single_layer = nn.Linear(ctx_dim, self.hidden_size)   
            if self.share_layer == None and self.deep_layer == None:  
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.num_layer - 1)
            elif self.share_layer != None and self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - self.share_layer[1] - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            elif self.share_layer != None and self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.share_layer[1], self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            else:
                raise ValueError('deep layer and share layer are not compatible!')
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors)
                
        print('Prompt design: SRDA for UDA')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of PDA context words (tokens): {n_ctx}")
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  
        self.name_lens = name_lens
        self.attn_block = CCDA_Module(clip_model, n_cls ,beta_s=0.1, beta_t=0.1)
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]


    def update_feature_bank(self, features, labels, max_probs, source=True, epoch=None):
        """
        Args:
            features: weak features (N, dim)
            labels: 标签或伪标签 (N,)
            max_probs: 最大预测概率 (N,)
            source: 是否为源域
            epoch: 当前epoch
        """
        if source:
            feat_bank = self.source_feat_bank
            max_prob_list = self.source_max_probs
            bank_counts=self.source_bank_counts
            threshold = self.w_s
        else:
            feat_bank = self.target_feat_bank
            max_prob_list = self.target_max_probs
            bank_counts=self.target_bank_counts
            # 动态衰减阈值
            threshold = self.w_low + (epoch / self.total_epochs) * (self.w_high - self.w_low)


        for i, (feat, label, prob) in enumerate(zip(features, labels, max_probs)):
            label = label.item()
            prob = prob.item()
            bank=feat_bank[label]
            probs=max_prob_list[label]
            count=bank_counts[label].item()

            if prob>=threshold:
                if(count<self.Q):
                    bank[count]=feat
                    probs[count]=prob
                    bank_counts[label]+=1
                else:
                    min_prob,min_idx=torch.min(probs,dim=0)
                    if prob>min_prob.item():
                        bank[min_idx]=feat
                        probs[min_idx]=prob

        self.bank_updates+=len(features)


    def update_feature_centers(self):
        """更新特征中心（动量更新）"""
        for k in range(self.n_cls):
            # Source centers update
            source_features = self.source_feat_bank[k]
            source_center_new = torch.mean(source_features, dim=0)
            self.source_center[k] = self.alpha * self.source_center[k] + (1 - self.alpha) * source_center_new

            # Target centers update
            target_features = self.target_feat_bank[k]
            if len(target_features) > 0:  # 确保有特征
                target_center_new = torch.mean(target_features, dim=0)
                self.target_center[k] = self.alpha * self.target_center[k] + (1 - self.alpha) * target_center_new

    def forward(self):
        if self.proj != None:
            vctx = self.proj(self.ctx)
        else:
            vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        if self.deep_share:
            deep_vctx = []
            for index, layer in enumerate(self.deep_prompt_proj):
                deep_vctx.append(layer(self.deep_ctx[index]))
            deep_vctx = torch.stack(deep_vctx)
            deep_vctx = torch.cat((deep_vctx, self.deep_vctx), dim=0)
        else:
            deep_vctx = self.deep_vctx
            
        return prompts, self.deep_ctx, vctx, deep_vctx


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K = self.prompt_learner.K
        self.dim = clip_model.text_projection.shape[1]
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        
        self.n_cls = len(classnames)
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.text_encoder_u = Simple_TextEncoder(clip_model)
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)
        

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.confi = cfg.CONFI
        self.epoch = cfg.EPOCH
        self.warm_up = cfg.WARM_UP

        self.temperature = 0.2
        self.name=cfg.DATASET.NAME



    def forward(self, image, label=None, epoch=None, train=False, construct=False, source=True,image_aug=False):
        prompts, deep_ctx, vctx, deep_vctx = self.prompt_learner()
                
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        text_features_u = self.text_encoder_u(self.tokenized_prompts_u.to(self.logit_scale.device))
        text_features_u = text_features_u / text_features_u.norm(dim=-1, keepdim=True)
        
        F_u = self.image_encoder(image.type(self.dtype), None, None)
        F_u = F_u / F_u.norm(dim=-1, keepdim=True)

        logits_u = logit_scale * F_u @ text_features_u.t()
        pseudo_label = torch.softmax(logits_u, dim=-1)
        max_probs, label_p = torch.max(pseudo_label, dim=-1)


        if train:
            self.prompt_learner.update_feature_bank(
                F_u,
                label if source else label_p,
                max_probs,
                source=source,
                epoch=epoch
            )

            # 计算特征中心
            source_center = self.prompt_learner.source_center
            target_center = self.prompt_learner.target_center
            logits_c = self.prompt_learner.attn_block(
                text_features,
                image_features,
                source_center,
                target_center
            )
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
            if source:  # 源域
                loss_x = F.cross_entropy(logits, label)
                if image_aug:
                    loss_c = 0
                else:
                    loss_c = F.cross_entropy(logits_c, label)
                return loss_x,loss_c,text_features, fixed_embeddings, zero_shot_features, image_features, zero_shot_logits, logits
            else:  # 目标域
                # 生成伪标签
                if epoch == None or epoch <= self.epoch:
                    logits_u = logit_scale * F_u @ text_features_u.t()
                    pseudo_label = torch.softmax(logits_u, dim=-1)
                else:
                    pseudo_label = torch.softmax(logits, dim=-1)
                max_probs, label_p = torch.max(pseudo_label, dim=-1)

                mask = max_probs.ge(self.confi).float()
                if mask.sum() == 0 or self.warm_up > epoch:
                    loss_c = torch.tensor(0.)
                    loss_x = torch.tensor(0.)
                else:
                    loss_c = (F.cross_entropy(logits_c, label_p, reduction="none") * mask).sum() / mask.sum()
                    loss_x = (F.cross_entropy(logits, label_p, reduction="none") * mask).sum() / mask.sum()
                if image_aug:
                    loss_c = 0
                else:
                    loss_c = loss_c

                return loss_x,loss_c, text_features, fixed_embeddings, zero_shot_features, image_features, zero_shot_logits, logits

        
        else:

            # 测试时使用特征中心
            source_center = self.prompt_learner.source_center
            target_center = self.prompt_learner.target_center
            logits_c = self.prompt_learner.attn_block(
                text_features,
                image_features,
                source_center,
                target_center
            )

            return logits + 0.5*logits_c


@TRAINER_REGISTRY.register()
class SRDA(BaseDA):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SRDA.PREC == "fp32" or cfg.TRAINER.SRDA.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.SRDA.PREC == "amp" else None

        self.step_counter = 1
        N = self.cfg.OPTIM.MAX_EPOCH
        mean = self.cfg.TRAINER.SRDA.DDCCR.GPA_MEAN
        stdev = self.cfg.TRAINER.SRDA.DDCCR.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.previous_model_gpa = None


    def after_epoch(self):
        # 每个epoch结束后更新特征中心
        self.model.prompt_learner.update_feature_centers()

        super().after_epoch()


    def forward_backward(self, batch_x, batch_u):
        prec = self.cfg.TRAINER.SRDA.PREC
        image_x, image_x2,label, image_u ,image_u2= self.parse_batch_train(batch_x, batch_u)

        if prec == "amp":
            with autocast():
                # 源域数据
                loss_x1,lossxc1, normalized_text_features_x, zs_clip_text_embeddings_x, zs_image_embedd_x, image_ft_x, zero_shot_logits_x, logits_x = self.model(
                    image_x, label, epoch=self.epoch, train=True)
                loss_x2, lossxc2,normalized_text_features_x2, zs_clip_text_embeddings_x2, zs_image_embedd_x2, image_ft_x2, zero_shot_logits_x2, logits_x2 = self.model(
                    image_x2, label, epoch=self.epoch, train=True,image_aug=True)
                # 目标域数据
                loss_u1, lossxu1,normalized_text_features_u, zs_clip_text_embeddings_u, zs_image_embedd_u, image_ft_u, zero_shot_logits_u, logits_u = self.model(
                    image_u, epoch=self.epoch, train=True, source=False)
                loss_u2,lossxu2, normalized_text_features_u2, zs_clip_text_embeddings_u2, zs_image_embedd_u2, image_ft_u2, zero_shot_logits_u2, logits_u2 = self.model(
                    image_u2, epoch=self.epoch, train=True, source=False,image_aug=True)

                # 计算源域对比损失
                loss_scl_text_x = F.l1_loss(normalized_text_features_x, zs_clip_text_embeddings_x.cuda(),
                                            reduction='mean') * self.cfg.TRAINER.SRDA.PROMPTSRC.TEXT_LOSS_WEIGHT
                loss_scl_image_x = F.l1_loss(image_ft_x, zs_image_embedd_x.cuda(),
                                             reduction='mean') * self.cfg.TRAINER.SRDA.PROMPTSRC.IMAGE_LOSS_WEIGHT
                L_SCL_logits_x = F.kl_div(
                    F.log_softmax(logits_x / 1, dim=1),
                    F.log_softmax(zero_shot_logits_x / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits_x.numel()
                # loss_scl_text_x2 = F.l1_loss(normalized_text_features_x2, zs_clip_text_embeddings_x2.cuda(),
                #                             reduction='mean') * self.cfg.TRAINER.PDA.DDCCR.TEXT_LOSS_WEIGHT
                loss_scl_image_x2 = F.l1_loss(image_ft_x2, zs_image_embedd_x2.cuda(),
                                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
                L_SCL_logits_x2 = F.kl_div(
                    F.log_softmax(logits_x2 / 1, dim=1),
                    F.log_softmax(zero_shot_logits_x2 / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits_x2.numel()

                # 计算目标域对比损失
                # loss_scl_text_u = F.l1_loss(normalized_text_features_u, zs_clip_text_embeddings_u.cuda(),
                #                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
                loss_scl_image_u = F.l1_loss(image_ft_u, zs_image_embedd_u.cuda(),
                                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
                L_SCL_logits_u = F.kl_div(
                    F.log_softmax(logits_u / 1, dim=1),
                    F.log_softmax(zero_shot_logits_u / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits_u.numel()

                # loss_scl_text_u2 = F.l1_loss(normalized_text_features_u2, zs_clip_text_embeddings_u2.cuda(),
                #                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
                loss_scl_image_u2 = F.l1_loss(image_ft_u2, zs_image_embedd_u2.cuda(),
                                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
                L_SCL_logits_u2 = F.kl_div(
                    F.log_softmax(logits_u2 / 1, dim=1),
                    F.log_softmax(zero_shot_logits_u2 / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logits_u2.numel()

                # # 总对比损失
                L_SCL_x = ( L_SCL_logits_x + L_SCL_logits_x2 + loss_scl_text_x + loss_scl_image_x + loss_scl_image_x2)
                L_SCL_u = ( L_SCL_logits_u + L_SCL_logits_u2  + loss_scl_image_u  + loss_scl_image_u2)

                total_loss = loss_x1  + loss_u1  + L_SCL_x + L_SCL_u+ loss_x2+ loss_u2+ lossxc1+ lossxu1


            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # 源域数据
            loss_x1, lossxc1,normalized_text_features_x, zs_clip_text_embeddings_x, zs_image_embedd_x, image_ft_x, zero_shot_logits_x, logits_x = self.model(
                image_x, label, epoch=self.epoch, train=True)
            loss_x2, lossxc2,normalized_text_features_x2, zs_clip_text_embeddings_x2, zs_image_embedd_x2, image_ft_x2, zero_shot_logits_x2, logits_x2 = self.model(
                image_x2, label, epoch=self.epoch, train=True,source=True,image_aug=True)

            # 目标域数据
            loss_u1,lossuc1, normalized_text_features_u, zs_clip_text_embeddings_u, zs_image_embedd_u, image_ft_u, zero_shot_logits_u, logits_u = self.model(
                image_u, epoch=self.epoch, train=True, source=False)
            loss_u2,lossuc2, normalized_text_features_u2, zs_clip_text_embeddings_u2, zs_image_embedd_u2, image_ft_u2, zero_shot_logits_u2, logits_u2 = self.model(
                image_u2, epoch=self.epoch, train=True, source=False,image_aug=True)

            # 计算源域对比损失
            loss_scl_text_x = F.l1_loss(normalized_text_features_x, zs_clip_text_embeddings_x.cuda(),
                                        reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
            loss_scl_image_x = F.l1_loss(image_ft_x, zs_image_embedd_x.cuda(),
                                         reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
            L_SCL_logits_x = F.kl_div(
                F.log_softmax(logits_x / 1, dim=1),
                F.log_softmax(zero_shot_logits_x / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits_x.numel()

            # loss_scl_text_x2 = F.l1_loss(normalized_text_features_x2, zs_clip_text_embeddings_x2.cuda(),
            #                              reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
            loss_scl_image_x2 = F.l1_loss(image_ft_x2, zs_image_embedd_x2.cuda(),
                                          reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
            L_SCL_logits_x2 = F.kl_div(
                F.log_softmax(logits_x2 / 1, dim=1),
                F.log_softmax(zero_shot_logits_x2 / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits_x2.numel()

            # 计算目标域对比损失
            # loss_scl_text_u = F.l1_loss(normalized_text_features_u, zs_clip_text_embeddings_u.cuda(),
            #                             reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
            loss_scl_image_u = F.l1_loss(image_ft_u, zs_image_embedd_u.cuda(),
                                         reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
            L_SCL_logits_u = F.kl_div(
                F.log_softmax(logits_u / 1, dim=1),
                F.log_softmax(zero_shot_logits_u / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits_u.numel()

            # loss_scl_text_u2 = F.l1_loss(normalized_text_features_u2, zs_clip_text_embeddings_u2.cuda(),
            #                              reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT
            loss_scl_image_u2 = F.l1_loss(image_ft_u2, zs_image_embedd_u2.cuda(),
                                          reduction='mean') * self.cfg.TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT
            L_SCL_logits_u2 = F.kl_div(
                F.log_softmax(logits_u2 / 1, dim=1),
                F.log_softmax(zero_shot_logits_u2 / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits_u2.numel()

            # 总对比损失
            L_SCL_x = (L_SCL_logits_x + L_SCL_logits_x2+ loss_scl_text_x + loss_scl_image_x  + loss_scl_image_x2)
            L_SCL_u = ( L_SCL_logits_u+L_SCL_logits_u2 + loss_scl_image_u + loss_scl_image_u2)

            total_loss = loss_x1  + loss_u1  + L_SCL_x + L_SCL_u+ loss_x2+ loss_u2+ lossuc1 +lossxc1
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            # 损失摘要
        loss_summary = {
            "loss": total_loss.item(),
            "loss_x": loss_x1.item(),
            "loss_u": loss_u1.item(),
            "acc_x": compute_accuracy(logits_x, label)[0].item(),
            "loss_scl_text_x": loss_scl_text_x.item(),
            "loss_scl_image_x": loss_scl_image_x.item(),
            "L_SCL_logits_x": L_SCL_logits_x.item(),
            "loss_scl_image_u": loss_scl_image_u.item(),
            "L_SCL_logits_u": L_SCL_logits_u.item(),
            "loss_scl_image_x2": loss_scl_image_x.item(),
            "L_SCL_logits_x2": L_SCL_logits_x.item(),
            "loss_scl_image_u2": loss_scl_image_u.item(),
            "L_SCL_logits_u2": L_SCL_logits_u.item(),
        }

        # 如果当前批次是最后一个批次，则更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            self.step_counter += 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(self.model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

            if self.step_counter == self.epoch + 1:
                self.model.load_state_dict(self.previous_model_gpa)

        return loss_summary
    
    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        # label_u is used only for evaluating pseudo labels' accuracy

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2


    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        modified_dict = dict2
        if not prompt_only:
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss
    

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
             
            
