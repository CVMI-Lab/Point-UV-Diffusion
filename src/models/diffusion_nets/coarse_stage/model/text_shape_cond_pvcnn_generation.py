import functools
import pdb
import torch.nn as nn
import torch
import numpy as np
import clip
from .cond_pvcnn_generation import Cond_PVCNN2 as BaseNet

class UV_PointNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UV_PointNet, self).__init__()
        self.fc1 = nn.Conv1d(in_ch, 64, 1)
        self.fc2 = nn.Conv1d(64, 128, 1)
        self.fc3 = nn.Conv1d(128, 256, 1)
        self.relu = nn.ReLU()
        self.fc_final = nn.Conv1d(512, out_ch, 1)

    def forward(self, x):
        b,c,h,w=x.size()
        x_flat = x.view(b,c,h*w)
        x_flat = self.relu(self.fc1(x_flat))
        x_flat = self.relu(self.fc2(x_flat))
        x_glob_feat = self.fc3(x_flat)
        x_feat_max = torch.max(x_glob_feat, -1, keepdim=True)[0]
        x_feat_avg = torch.mean(x_glob_feat, -1, keepdim=True)
        x_feat = torch.cat([x_feat_max, x_feat_avg], dim=1)
        out = self.fc_final(x_feat)
        return out

class Condition_Net(BaseNet):
    def __init__(self, text_embed_ch, cond_embed_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_encoder = UV_PointNet(7, cond_embed_ch)
        self.embedding_size = 768
        self.fc1 = nn.Linear(self.embedding_size, 256)
        self.fc2 = nn.Linear(256, text_embed_ch)
        self.silu= nn.SiLU()

    def forward(self, x, t, fps_cond=None, shape_cond=None, text=None, **kwargs):
        assert shape_cond.size()[1] == 7
        shape_feat = self.shape_encoder(shape_cond)
        cond = torch.cat([fps_cond, shape_feat.repeat(1, 1, self.fps_num)], dim=1)

        inputs = x
        temb = self.embedf(self.get_timestep_embedding(t, inputs.device))[:, :, None].expand(-1, -1, inputs.shape[-1])

        if text!=None:
            text_embed = text.float()
            text_embed = self.silu(self.fc1(text_embed))
            text_embed = self.fc2(text_embed)

            temb = temb + text_embed.unsqueeze(2).expand(text_embed.shape[0], text_embed.shape[1], inputs.shape[-1])


        coords, features = cond[:, :3, :].contiguous(), torch.cat([inputs, cond], dim=1)
        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))

        in_features_list[0] = cond.contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((coords_list[-1 - fp_idx], coords, torch.cat([features, temb], dim=1),
                                                in_features_list[-1 - fp_idx], temb))

        out = self.classifier(features)

        return out
