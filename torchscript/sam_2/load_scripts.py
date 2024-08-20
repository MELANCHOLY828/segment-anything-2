import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from PIL import Image
import numpy as np
# from torchvision.transforms import Normalize, Resize, ToTensor

class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution = 1024, mask_threshold = 0.0, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        from torchvision.transforms import Normalize, Resize, ToTensor
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
    def forward(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

image = Image.open('E:/work/data/my/toy/images/frame_0000.jpg')
image = np.array(image.convert("RGB"))
H = image.shape[0]
W = image.shape[1]
img_size = 1024
transform = SAM2Transforms()
image = transform(image)
img = image[None, ...]
# import pdb
# pdb.set_trace() 
sam2_encoder = torch.jit.load("E:/work/code/script/sam2_hiera_base_plus_encoder.pt")
# img=torch.randn(1, 3, 1024, 1024)
high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)
# embed_dim = sam2_encoder.sam_prompt_encoder.embed_dim
# embed_size = (sam2_encoder.image_size // sam2_encoder.backbone_stride, sam2_encoder.image_size // sam2_encoder.backbone_stride)
mask_input_size = torch.rand(256,256)

sam2_decoder = torch.jit.load("E:/work/code/script/sam2_hiera_base_plus_decoder.pt")
# point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float)
# point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float)

point_coords = torch.tensor(([285, 360], [412, 512])).unsqueeze(0)
point_coords[...,0] = point_coords[...,0] / W * img_size
point_coords[...,1] = point_coords[...,1] / H * img_size

point_labels = torch.tensor([1,1]).unsqueeze(0)

mask_input = torch.randn(1, 1, 256, 256, dtype=torch.float)
has_mask_input = torch.tensor([1], dtype=torch.float)
orig_im_size = torch.tensor([1024, 1024], dtype=torch.int32)
masks, scores = sam2_decoder(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input, orig_im_size)
masks = F.interpolate(masks, (H,W), mode="bilinear", align_corners=False)
mask = masks < 0.0
mask = mask.squeeze(0)
mask = mask.expand(3, -1, -1)
mask = mask.numpy()
mask = (mask * 255.).astype(np.uint8)
mask = np.transpose(mask, (1, 2, 0))
image = Image.fromarray(mask)
image.save('mask_image.png')