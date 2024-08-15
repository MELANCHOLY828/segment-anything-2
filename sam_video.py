import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import os
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
import numpy as np
from PIL import Image
def getmask(video_dir):
    last_dir = os.path.join(os.path.dirname(video_dir), 'mask')
    if not os.path.exists(last_dir):
        os.makedirs(last_dir)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state, frame_names = predictor.init_state(video_dir)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        points = np.array([[270, 364], [419, 498]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 1], np.int32)

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        mask = masks[0] < 0
        mask = mask.squeeze(0)
        mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype('uint8'))
        mask_img.save(os.path.join(last_dir, frame_names[ann_frame_idx]))
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask = masks[0] < 0
            mask = mask.squeeze(0)
            mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype('uint8'))
            mask_img.save(os.path.join(last_dir, frame_names[frame_idx]))
getmask('E:/work/data/my/toy/images')