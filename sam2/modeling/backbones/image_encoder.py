# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]

        output = {
            "vision_features": src,
            "vision_pos_enc_0": pos[0],
            "vision_pos_enc_1": pos[1],
            "vision_pos_enc_2": pos[2],
            "backbone_fpn_0": features[0],
            "backbone_fpn_1": features[1],
            "backbone_fpn_2": features[2],
        }
        # assert isinstance(output["vision_features"], torch.Tensor)
        # assert isinstance(output["vision_pos_enc"], list) and all(isinstance(item, torch.Tensor) for item in output["vision_pos_enc"]), "vision_pos_enc should be a list of Tensors"
        # assert isinstance(output["backbone_fpn"], list) and all(isinstance(item, torch.Tensor) for item in output["backbone_fpn"]), "backbone_fpn should be a list of Tensors"
        return output
# add ModuleInterface 
@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass

class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [torch.zeros_like(xs[0])] * len(self.convs)
        pos = [torch.zeros_like(xs[0])] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        # prev_features = None
        prev_features = torch.zeros_like(xs[0])
        # forward in top-down order (from low to high resolution)
        flag = False

        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            # x = torch.gather(xs, 0, torch.tensor([i]))
            x = xs[i]
            submodule : ModuleInterface = self.convs[n - i]
            lateral_features = submodule.forward(x)
            # lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and flag:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
                flag = True
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
        # n = len(self.convs) - 1
        # out = [None] * len(xs)
        # pos = [None] * len(xs)
        # prev_features = None
        # for i, x in enumerate((xs)):  # 使用 enumerate 获取索引和元素
        #     lateral_features = self.convs[n - i](x)  # 使用索引访问 self.convs
        #     if i in self.fpn_top_down_levels and prev_features is not None:
        #         top_down_features = F.interpolate(
        #             prev_features.to(dtype=torch.float32),
        #             scale_factor=2.0,
        #             mode=self.fpn_interp_model,
        #             align_corners=(
        #                 None if self.fpn_interp_model == "nearest" else False
        #             ),
        #             antialias=False,
        #         )
        #         prev_features = lateral_features + top_down_features
        #         if self.fuse_type == "avg":
        #             prev_features /= 2
        #     else:
        #         prev_features = lateral_features
        #     x_out = prev_features
        #     out[i] = x_out
        #     pos[i] = self.position_encoding(x_out).to(x_out.dtype)
        # return out, pos