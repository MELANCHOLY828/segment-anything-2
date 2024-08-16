import onnx
from onnx import numpy_helper
import torch
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageApp:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.click_count = 0
        self.coors = []
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<Button-1>", self.get_pixel)

    def get_pixel(self, event):
        x, y = event.x, event.y
        pixel = self.image.getpixel((x, y))
        self.coors.append(x)
        self.coors.append(y)
        self.click_count += 1
        if self.click_count == 2:
            self.root.quit()

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
predictor.eval()
video_dir = 'E:/work/data/my/toy/images'
last_dir = os.path.join(os.path.dirname(video_dir), 'mask')
if not os.path.exists(last_dir):
    os.makedirs(last_dir)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    root = tk.Tk()
    app = ImageApp(root, os.path.join(video_dir, 'frame_0000.jpg'))
    root.mainloop()
    coors = app.coors
    app.root.destroy()
torch.onnx.export(predictor,
(video_dir, last_dir, coors),
"E:/work/code/SAM2.onnx",
verbose=True,
input_names=['video_dir', 'last_dir', 'coors'],
output_names=['masks'],
# dynamic_axes={'points': {0: 'num_points'}, 'masks': {0: 'num_frames'}}
)
model_onnx = onnx.load(r"E:/work/code/SAM2.onnx")                   # onnx加载保存的onnx模型
onnx.checker.check_model(model_onnx) 
print(onnx.helper.printable_graph(model_onnx.graph)) 
# def getmask(video_dir):
#     last_dir = os.path.join(os.path.dirname(video_dir), 'mask')
#     if not os.path.exists(last_dir):
#         os.makedirs(last_dir)
#     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#         root = tk.Tk()
#         app = ImageApp(root, os.path.join(video_dir, 'frame_0000.jpg'))
#         root.mainloop()
#         coors = app.coors
#         predictor(video_dir, last_dir, coors)
# getmask('E:/work/data/my/toy/images')