"""Vision backbones for tinydp's DiffusionPolicy, without editing the tinydp submodule (policy environment).

tinydp's DiffusionPolicy builds `get_resnet('resnet18')`: a ResNet18 from random weights (BatchNorm -> GroupNorm),
trained end to end, with global average pooling into its 512 features. Average pooling discards most of where
things are in the image. The OpenArm cup policy at 20k steps predicted the start of the approach 6-7 cm off (the
cup's position is only in the image there) while mid-carry it was 1 cm off, and closed-loop it never reached the
handle. Diffusion Policy (Chi et al.) and robomimic replace the pooling with a spatial softmax: per keypoint channel
the expected image position of its activation, which keeps object positions.

  resnet18_avgpool          tinydp's own backbone (no patch)
  resnet18_spatial_softmax  ResNet18 (ImageNet weights if pretrained) with avgpool replaced by a 1x1 conv to 256
                            keypoint maps + spatial softmax -> 256 x (x, y) = 512 features (DiffusionPolicy's
                            vision_feature_dim, so nothing else changes)

Image crop (arch["image_crop"], x0 y0 w h in the policy's camera frame): the cup and the hand only use part of the
1280x1080 robot camera image; cropping before tinydp's resize puts more pixels on the cup (~19 px wide at 320 px
from the full frame). apply_crop() patches tinydp's default / augmenting vision transforms to crop first;
with_crop() adds the crop to a transform directly (OpenArmCupPolicy).

apply(arch) patches tinydp.policy.model.get_resnet before a DiffusionPolicy is built, and the transforms.
train_tinydp.py applies it for training and writes the arch into the run folder; export_policy.py puts it into the
policy file and OpenArmCupPolicy applies it before loading (pretrained=False there: the weights come from the file).
"""

import torch
import torch.nn as nn
import torchvision

ARCHS = ("resnet18_avgpool", "resnet18_spatial_softmax")


class SpatialSoftmax(nn.Module):
    """B, C, H, W feature map -> B, 2K: the softmax-weighted mean (x, y) in [-1, 1] of each of K keypoint maps."""

    def __init__(self, channels: int, keypoints: int):
        super().__init__()
        self.keypoints = nn.Conv2d(channels, keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.keypoints(x)
        _, _, h, w = x.shape
        attention = torch.softmax(x.flatten(2), dim=-1)  # B, K, H*W
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                                torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing="ij")
        return torch.cat([(attention * xs.flatten()).sum(-1), (attention * ys.flatten()).sum(-1)], dim=-1)


def resnet18_spatial_softmax(pretrained: bool) -> nn.Module:
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    resnet.avgpool = SpatialSoftmax(512, 256)  # torchvision's forward then flattens (a no-op) and applies fc
    resnet.fc = nn.Identity()
    return resnet


class ImageCrop:
    """PIL image -> its (x0, y0, w, h) window (a class, so tinydp's data workers can unpickle it)."""

    def __init__(self, xywh):
        self.xywh = tuple(int(v) for v in xywh)

    def __call__(self, image):
        x0, y0, w, h = self.xywh
        return image.crop((x0, y0, x0 + w, y0 + h))

    def __repr__(self) -> str:
        return f"ImageCrop(xywh={self.xywh})"


def with_crop(transform, crop):
    """tinydp vision transform (torchvision Compose of PIL transforms) -> the same with the crop first (crop: None
    or x0 y0 w h)."""
    if crop:
        transform.transforms.insert(0, ImageCrop(crop))
    return transform


def apply_crop(crop) -> None:
    """Make tinydp.dataset.utils' default_vision_transform / aug_vision_transform crop first (None: tinydp's own)."""
    import tinydp.dataset.utils as tinydp_utils

    originals = {}
    for name in ("default_vision_transform", "aug_vision_transform"):
        originals[name] = getattr(tinydp_utils, f"_tinydp_{name}", getattr(tinydp_utils, name))
        setattr(tinydp_utils, f"_tinydp_{name}", originals[name])
    for name, original in originals.items():
        setattr(tinydp_utils, name, original if not crop else
                (lambda original: lambda *a, **kw: with_crop(original(*a, **kw), crop))(original))


def apply(arch: dict, pretrained: bool | None = None) -> None:
    """arch: {"backbone": one of ARCHS, "pretrained": bool}; pretrained overrides arch's (False when loading weights)."""
    import tinydp.policy.model as tinydp_model

    if arch["backbone"] not in ARCHS:
        raise ValueError(f"unknown vision backbone {arch['backbone']!r} (known: {ARCHS})")
    apply_crop(arch.get("image_crop"))
    original = getattr(tinydp_model, "_tinydp_get_resnet", tinydp_model.get_resnet)
    tinydp_model._tinydp_get_resnet = original  # so a later apply (another policy file) can restore it
    if arch["backbone"] == "resnet18_avgpool":
        tinydp_model.get_resnet = original
        return
    use_pretrained = arch.get("pretrained", False) if pretrained is None else pretrained
    tinydp_model.get_resnet = lambda name, weights=None, **kwargs: resnet18_spatial_softmax(use_pretrained)
