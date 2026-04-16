import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask


class FootEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        max_len=120,
        # condition
        cliffcam_dim=3,
        # intermediate
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2.0,
        # training
        dropout=0.1,
    ):
        super().__init__()
        self.num_2d_joints = 2 * 4  # 4 joints for left and right foot
        self.num_rot_condition = 4  # condition on left and right knee and initial ankle
        self.output_dim = 2 * 6  # left and right ankle joint in 6D representation
        self.max_len = max_len
        self.cliffcam_dim = cliffcam_dim
        assert self.cliffcam_dim > 0

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(
            torch.randn(self.num_2d_joints, 32), requires_grad=True
        )
        self.embed_noisyobs = Mlp(
            self.num_2d_joints * 32,
            hidden_features=self.latent_dim * 2,
            out_features=self.latent_dim,
            drop=dropout,
        )

        self._build_condition_embedder()

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        self.rot6d_embedder = nn.Sequential(
            nn.Linear(6 * self.num_rot_condition, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, length, obs, f_cliffcam, global_rot6d):
        B, L, J, C = obs.shape
        assert C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = []
        f_to_add.append(self.cliffcam_embedder(f_cliffcam))
        f_to_add.append(self.rot6d_embedder(global_rot6d))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        # predict residual to initial ankle rotations
        sample = sample + global_rot6d[:, :, -12:]
        return sample
