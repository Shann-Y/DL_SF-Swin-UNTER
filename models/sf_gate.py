#!/usr/bin/env python3
"""Spatial-Frequency Gate (SF-Gate) module.

Introduces frequency-domain feature selection into skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SFGate3D(nn.Module):
    """3D Spatial-Frequency Gate.

    Fuses spatial (channel attention) and frequency (FFT) features with a
    learned gate.
    """

    def __init__(self, channels, reduction=16, spatial_size=None):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # spatial branch: channel attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
        )

        # frequency branch: FFT -> [Re; Im] -> 1x1 conv
        self.freq_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
        )

        # gated fusion
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Conv3d(channels, channels, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_feat, decoder_feat):
        # align spatial sizes if needed
        if encoder_feat.shape != decoder_feat.shape:
            decoder_feat = F.interpolate(
                decoder_feat,
                size=encoder_feat.shape[2:],
                mode='trilinear',
                align_corners=False,
            )

        # spatial branch
        spatial_att = self.spatial_attention(encoder_feat)
        spatial_feat = encoder_feat * torch.sigmoid(spatial_att)

        # frequency branch. Forced float32 under AMP: the FFT DC component is
        # O(num voxels) and overflows fp16 -> NaN.
        with torch.autocast(device_type="cuda", enabled=False):
            freq_feat = self._frequency_transform(encoder_feat)
            freq_att = self.freq_conv(freq_feat)
            freq_feat = encoder_feat.float() * torch.sigmoid(freq_att)
        freq_feat = freq_feat.to(encoder_feat.dtype)

        # gated fusion
        concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)
        gate_weight = self.gate(concat_feat)
        fused_feat = gate_weight * spatial_feat + (1 - gate_weight) * freq_feat

        # combine with the decoder feature (residual)
        output = self.output_proj(fused_feat) + decoder_feat

        return output

    def _frequency_transform(self, x):
        """3D FFT -> fftshift -> concat(real, imag), always in float32."""
        x_float = x.float()
        freq = torch.fft.fftn(x_float, dim=(-3, -2, -1))
        freq_shifted = torch.fft.fftshift(freq, dim=(-3, -2, -1))
        return torch.cat([freq_shifted.real, freq_shifted.imag], dim=1)


class SFGateBlock(nn.Module):
    """SF-Gate with a residual connection."""

    def __init__(self, channels, reduction=16, use_residual=True):
        super().__init__()
        self.sf_gate = SFGate3D(channels, reduction)
        self.use_residual = use_residual
        if use_residual:
            self.residual_conv = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, encoder_feat, decoder_feat):
        out = self.sf_gate(encoder_feat, decoder_feat)
        if self.use_residual:
            out = out + self.residual_conv(decoder_feat)
        return out
