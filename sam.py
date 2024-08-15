import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
image = Image.open('E:/work/code/segment-anything-2/images/truck.jpg')
image = np.array(image.convert("RGB"))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
    masks = (masks * 255.).astype(np.uint8)
    masks = np.transpose(masks, (1, 2, 0))
    image = Image.fromarray(masks)
    image.save('mask_image.png')
    print("ok")