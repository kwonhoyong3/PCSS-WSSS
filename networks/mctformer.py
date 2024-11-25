import torch
import torch.nn as nn
from functools import partial
from networks.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from networks.vision_transformer import Block, PatchEmbed
from torch.cuda.amp import autocast

import math

__all__ = ['deit_small_MCTformerV2_PCSS_patch16_224']
      
class MCTformerV2_PCSS(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_tokenizer = False

        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches

        if self.is_tokenizer:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.tokenizer =  nn.Linear(384, 384*self.num_classes,bias=True)
        else: 
            self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))


        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, merge_block=None, mag=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if self.is_tokenizer:
            cls_fg = self.cls_token.expand(B, -1,-1)
            cls_tokens = self.tokenizer(cls_fg.view(B,-1)).view(B,self.num_classes,self.embed_dim)  # tokenizer: Residual
            cls_tokens[:,:,:]+=cls_fg.view(B,1,self.embed_dim)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        x_patch_list = []
        ctk_list = []

        _, p, D = x.shape
        p = p - self.num_classes

        for i, blk in enumerate(self.blocks):
            if merge_block != None and mag != None and i == (merge_block+1):
                x_c = x[:,:self.num_classes]
                x_p = x[:,self.num_classes:]
                x_p = torch.reshape(x_p, [B, int(p ** 0.5), int(p ** 0.5), D])
                x_p = x_p.permute([0, 3, 1, 2])     # B D W H
                
                x_fft = torch.fft.fftn(x_p, dim=(2,3))

                x_pha = torch.angle(x_fft)
                x_mag = torch.abs(x_fft)

                x_fft_mag_merge = mag * torch.exp((1j) * x_pha)

                x_p = torch.fft.ifftn(x_fft_mag_merge, dim=(2,3)).real    # B D W H
                x_p = x_p.permute([0, 2, 3, 1]).reshape(B, -1, D) # B WH D

                x = torch.cat((x_c, x_p), dim=1)
            
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

            ctk_list.append(x[:, 0:self.num_classes])
            if merge_block != None and i == merge_block:
                x_patch_list.append(x[:, self.num_classes:])

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, x_patch_list, ctk_list

    def forward(self, x, merge_block=None, mag=None, is_freq=False, return_att=False, n_layers=12):
        w, h = x.shape[2:]
        
        x_cls, x_patch, attn_weights, x_patch_list, ctk_list = self.forward_features(x, merge_block, mag)

        n, p, D = x_patch.shape

        x_cls_logits = x_cls.mean(-1)   

        for i in range(len(x_patch_list)):
            xp = x_patch_list[i]
            if w != h:
                w0 = w // self.patch_embed.patch_size[0]
                h0 = h // self.patch_embed.patch_size[0]
                xp = torch.reshape(xp, [n, w0, h0, D])
                #########################
            else:
                xp = torch.reshape(xp, [n, int(p ** 0.5), int(p ** 0.5), D])
                #########################
            xp = xp.permute([0, 3, 1, 2])
            xp = xp.contiguous()

            x_patch_list[i] = xp      

        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, D])
            #########################
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), D])
            #########################
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        # feat = x_patch
        
        x_patch = self.head(x_patch)

        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) ######ORIGINAL
        
        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        x_patch = F.relu(x_patch)

        np, cp, hp, wp = x_patch.shape

        mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([np, cp, hp, wp])

        cams = mtatt * x_patch  # B * C * 14 * 14

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]
        patch_attn = torch.sum(patch_attn, dim=0) #B 196 196

        rcams = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],cams.shape[1], -1, 1)).reshape(cams.shape[0],cams.shape[1], hp, wp) #(B 1 N2 N2) * (B,20,N2,1)
        
        outs = {}
        outs['cls']= x_cls_logits
        outs['pcls']= x_patch_logits
        outs['cams']= x_patch
        outs['rcams']= rcams
        outs['x_patch_list'] = x_patch_list
        outs['ctk_list'] = ctk_list

        if is_freq:
            return x_cls_logits, x_patch_logits

        if return_att:
            return rcams
        else:
            return outs 

@register_model
def deit_small_MCTformerV2_PCSS_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV2_PCSS(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

