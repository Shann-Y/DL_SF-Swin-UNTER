#!/usr/bin/env python3
"""SF-Swin-UNETR: Swin-UNETR with a Spatial-Frequency Gate in the skip connections."""

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union
from functools import partial

from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock

from .sf_gate import SFGateBlock


class SFSwinUNETR(nn.Module):
    """Swin-UNETR with SF-Gate modules inserted into the skip connections.

    Args:
        img_size: input patch size.
        in_channels: input channels.
        out_channels: number of classes.
        feature_size: base feature dimension of the Swin backbone.
        use_checkpoint: enable gradient checkpointing.
        use_sf_gate: enable the SF-Gate modules.
        sf_gate_reduction: channel-reduction ratio inside the SF-Gate.
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        use_sf_gate: bool = True,
        sf_gate_reduction: int = 16,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.use_sf_gate = use_sf_gate
        self.feature_size = feature_size

        # MONAI removed the `img_size` argument from SwinUNETR in 1.5+, so pass
        # it only when the installed version still accepts it.
        swin_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
            swin_kwargs["img_size"] = img_size
        self.swin_unetr = SwinUNETR(**swin_kwargs)

        if use_sf_gate:
            # channel sizes of the four skip-connection features
            channel_sizes = [
                feature_size,      # enc0
                feature_size,      # enc1
                feature_size * 2,  # enc2
                feature_size * 4,  # enc3
            ]
            self.sf_gates = nn.ModuleList([
                SFGateBlock(ch, reduction=sf_gate_reduction)
                for ch in channel_sizes
            ])

        self._init_weights()

    def _init_weights(self):
        if self.use_sf_gate:
            for gate in self.sf_gates:
                for m in gate.modules():
                    if isinstance(m, nn.Conv3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def load_ssl_pretrained(self, weight_path: str):
        """Load MONAI's self-supervised Swin backbone weights (model_swinvit.pt).

        The checkpoint stores a DataParallel-wrapped Swin Transformer plus SSL
        task heads (rotation / contrastive / decoder). This strips the
        ``module.`` prefix, drops the SSL heads and the final LayerNorm, maps
        keys to ``swin_unetr.swinViT.*``, and renames ``mlp.fc*`` -> ``mlp.linear*``
        to match MONAI's MLPBlock.
        """
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        skip_prefixes = ('convTrans3d', 'rotation_head', 'contrastive_head')
        encoder_dict = {}
        for k, v in state_dict.items():
            k = k[len('module.'):] if k.startswith('module.') else k
            if k.startswith(skip_prefixes) or k in ('norm.weight', 'norm.bias'):
                continue
            new_key = 'swin_unetr.swinViT.' + k
            new_key = new_key.replace('mlp.fc1', 'mlp.linear1').replace('mlp.fc2', 'mlp.linear2')
            encoder_dict[new_key] = v

        missing, unexpected = self.load_state_dict(encoder_dict, strict=False)
        print(f"[SSL Pretrained] Loaded {len(encoder_dict)} backbone weights")
        print(f"[SSL Pretrained] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return missing, unexpected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_sf_gate:
            return self.swin_unetr(x)

        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)

        # encoder features used by the four skip connections
        enc0 = self.swin_unetr.encoder1(x)               # stem
        enc1 = self.swin_unetr.encoder2(hidden_states[0])
        enc2 = self.swin_unetr.encoder3(hidden_states[1])
        enc3 = self.swin_unetr.encoder4(hidden_states[2])

        dec4 = self.swin_unetr.encoder10(hidden_states[4])  # bottleneck
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states[3])

        # decoder upsampling, applying SF-Gate at each skip connection
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec2 = self.sf_gates[3](enc3, dec2)

        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec1 = self.sf_gates[2](enc2, dec1)

        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        dec0 = self.sf_gates[1](enc1, dec0)

        out_up = self.swin_unetr.decoder1(dec0, enc0)
        out = self.sf_gates[0](enc0, out_up)

        return self.swin_unetr.out(out)

    def get_sf_gate_params(self):
        """Parameters of the SF-Gate modules (for a separate learning rate)."""
        if self.use_sf_gate:
            return list(self.sf_gates.parameters())
        return []

    def get_backbone_params(self):
        """Parameters of the Swin-UNETR backbone."""
        return list(self.swin_unetr.parameters())


def create_model(
    img_size: Tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 1,
    out_channels: int = 2,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    use_sf_gate: bool = True,
    sf_gate_reduction: int = 16,
    pretrained_path: Optional[str] = None,
) -> SFSwinUNETR:
    """Build an SFSwinUNETR, optionally loading SSL pretrained weights."""
    model = SFSwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        use_sf_gate=use_sf_gate,
        sf_gate_reduction=sf_gate_reduction,
    )
    if pretrained_path is not None:
        model.load_ssl_pretrained(pretrained_path)
    return model
