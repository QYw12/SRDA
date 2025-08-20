from einops import rearrange
import torch.nn as nn
import torch
class CrossAttentionLayer(nn.Module):
    def __init__(self, pre_project, post_project, softmax, scale, logit_scale):
        super().__init__()
        self.pre_project = pre_project
        self.post_project = post_project
        self.softmax = softmax
        self.scale = scale
        self.logit_scale = logit_scale  # 添加 logit_scale 到构造函数

    def forward(self, Ft, Fv, Fvs_bank, Fvt_bank, beta_s=1.0, beta_t=1.0):
        
        Fvs_bank = Fvs_bank.to(self.pre_project[0].weight.dtype)
        Fvt_bank = Fvt_bank.to(self.pre_project[0].weight.dtype)
        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)
        out_fvt = self.pre_project(Fvt_bank)  # (N, 3 * C)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        q_fvt, k_fvt, v_fvt = tuple(rearrange(out_fvt, 'b (d k) -> k b d ', k=3))

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))  # (batch, N)

        Fsa = Fv + self.post_project(As @ v_fvs)  # (batch, C)
        Fta = Fv + self.post_project(At @ v_fvt)  # (batch, C)

        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        Fta = Fta / Fta.norm(dim=-1, keepdim=True)


        logit_scale_exp = self.logit_scale.exp()
        logits_s = beta_s * logit_scale_exp * Fsa @ Ft.t()
        logits_t = beta_t * logit_scale_exp * Fta @ Ft.t()
        logits = logits_s + logits_t

        return {
            'logits': logits,
            'Fsa': Fsa,
            'Fta': Fta,
            'Fvs': Fvs_bank,
            'Fvt': Fvt_bank
        }


class CCDA_Module(nn.Module):

    def __init__(self, clip_model, n_cls, beta_s=1.0, beta_t=1.0):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = clip_model.text_projection.shape[1]
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8

        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1
        self.n_cls = n_cls

        self.pre_project = nn.Sequential(
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim2, input_dim * 3)
        ).half()

        self.post_project = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        ).half()

        self.logit_scale = clip_model.logit_scale

        self.cross_attention_layer_1 = CrossAttentionLayer(self.pre_project, self.post_project, self.softmax,
                                                           self.scale, self.logit_scale)  # 传递 logit_scale
        self.cross_attention_layer_2 = CrossAttentionLayer(self.pre_project, self.post_project, self.softmax,
                                                           self.scale, self.logit_scale)  # 传递 logit_scale

    def forward(self, Ft, Fv, Fvs_bank, Fvt_bank):
        
        logits_first_layer = self.cross_attention_layer_1(Ft, Fv, Fvs_bank, Fvt_bank, self.beta_s, self.beta_t)

        
        Fv_fused = (logits_first_layer['Fsa'] + logits_first_layer['Fta'])//2
        logits_second_layer = self.cross_attention_layer_2(Ft, Fv_fused, logits_first_layer['Fvs'],
                                                           logits_first_layer['Fvt'], self.beta_s, self.beta_t)
        #
        
        logits = logits_first_layer['logits']+logits_second_layer['logits']

        return logits
