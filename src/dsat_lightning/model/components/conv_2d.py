"""Conv2d with same-padding."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_conv2d(
    x: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> list[int]:
    """Padding (left, right, top, bottom) so that out_size = ceil(in_size / stride)."""
    h, w = x.shape[2], x.shape[3]
    pad_h = max((math.ceil(h / stride[0]) - 1) * stride[0] + kernel_size[0] - h, 0)
    pad_w = max((math.ceil(w / stride[1]) - 1) * stride[1] + kernel_size[1] - w, 0)
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        s = stride if isinstance(stride, tuple) else (stride, stride)
        self.kernel_size = k
        self.stride = s
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = _same_pad_conv2d(x, self.kernel_size, self.stride)
        if any(pad):
            x = F.pad(x, pad)
        return self.conv(x)
