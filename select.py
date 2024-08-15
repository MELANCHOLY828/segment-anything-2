from PIL import Image, ImageTk
import tkinter as tk

class ImageApp:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<Button-1>", self.get_pixel)
        self.coors = []
        self.click_count = 0
    def get_pixel(self, event):
        
        x, y = event.x, event.y
        pixel = self.image.getpixel((x, y))
        print(f"Pixel at ({x}, {y}): {pixel}")
        self.coors.append(x)
        self.coors.append(y)
        self.click_count += 1
        if self.click_count == 2:
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root, 'E:/work/data/my/toy/images/frame_0000.jpg')
    root.mainloop()
    print(app.coors)
