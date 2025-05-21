import tkinter as tk
from tkinter import Canvas, Label, Frame, Button, filedialog
from PIL import Image, ImageTk
import numpy as np

# 定义画布长宽
width_canvas = 700
height_canvas = 700


def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


gui = tk.Tk()
gui.title("数字图像处理实验① 图像显示")

label = Label(gui, text="生成的图像", font=("楷体", 16))
label.pack(pady=10)

canvas = Canvas(gui, width=width_canvas, height=height_canvas, background=rgb_to_hex(80, 140, 100))
canvas.pack()

info_label = Label(gui, text="图像信息：", font=("楷体", 14))
info_label.pack(pady=5)


def display_image(image_array, is_color=False):
    if is_color:
        image = Image.fromarray(image_array, 'RGB')
    else:
        image = Image.fromarray(image_array, 'L')

    photo = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(width_canvas / 2, height_canvas / 2, anchor=tk.CENTER, image=photo)
    canvas.image = photo
    label.config(text="图像展示")
    info_label.config(text=f"图像信息：{image_array.shape}")


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image_array = np.array(image)
        display_image(image_array, is_color=len(image_array.shape) == 3)


def generate_black_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    display_image(image_array)


def generate_white_image():
    image_array = np.full((512, 512), 255, dtype=np.uint8)
    display_image(image_array)


def generate_gradient_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    for x in range(512):
        value = min(x, 255)
        image_array[:, x] = value
    display_image(image_array)


def generate_color_image():
    image_array = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        for y in range(512):
            red = min(0.5 * x, 255)
            green = min(0.5 * y, 255)
            blue = min((1 * x + 9 * y) / 10, 255)
            image_array[y, x] = [red, green, blue]
    display_image(image_array, is_color=True)


def generate_stripes_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    values = [0, 31, 63, 95, 127, 159, 191, 224, 255]
    width = 512 // len(values)
    for i in range(1, len(values) + 1):
        value = values[i - 1]
        start = i * width
        end = (i + 1) * width
        image_array[:, start:end] = value
    display_image(image_array)


button_frame = Frame(gui)
button_frame.pack(pady=20)

buttons = [
    ("全黑图像", generate_black_image),
    ("全白图像", generate_white_image),
    ("渐变图像", generate_gradient_image),
    ("彩色图像", generate_color_image),
    ("条纹图像", generate_stripes_image),
    ("打开文件", open_image)
]

for i, (text, command) in enumerate(buttons):
    button = Button(button_frame, text=text, command=command, font=("楷体", 18))
    button.grid(row=0, column=i, padx=15, pady=10)

gui.mainloop()
