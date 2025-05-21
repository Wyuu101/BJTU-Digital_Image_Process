import tkinter as tk
from tkinter import Canvas, Label, Frame, Button, filedialog
from PIL import Image, ImageTk
import numpy as np
import struct
import os
import re

# 定义画布长宽
width_canvas = 600
height_canvas = 600
#全局变量（原始文件名称列表、原始文件目录、当前图像索引）
global raw_files, dir_path, current_index

#把rgb数值转化为16进制
def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'
#正则匹配表达式
def extract_parts(file_name):
    # 提取字母部分和数字部分
    match = re.match(r'([a-zA-Z]+)(\d*)', file_name)
    letters = match.group(1)
    number = int(match.group(2)) if match.group(2) else float('inf')  # 如果没有数字，设置为inf
    return (letters, number)

#显示图片
def display_image(image_array, is_color=False,file_name=None):
    if is_color:
        image = Image.fromarray(image_array, 'RGB')
    else:
        image = Image.fromarray(image_array, 'L')

    photo = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(width_canvas / 2, height_canvas / 2, anchor=tk.CENTER, image=photo)
    canvas.image = photo
    info_name.config(text=f"名称：{file_name}")
    info_height.config(text=f"高：{image_array.shape[0]}")
    info_weight.config(text=f"宽：{image_array.shape[1]}")
    if len(image_array.shape) >2:
        info_color.config(text=f"通道：{image_array.shape[2]}")
    else:
        info_color.config(text=f"通道：1")

#打开指定图片
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image_array = np.array(image)
        display_image(image_array, is_color=len(image_array.shape) == 3,file_name=os.path.basename(file_path))
        clear_tips()

#读取原始图像数据
def read_raw(file_name):
    with open(file_name, 'rb') as f:
        weight, height = struct.unpack("II", f.read(8))
        data_size = weight * height
        raw_data = struct.unpack(f"{data_size}B", f.read(data_size))
        return np.array(raw_data, np.uint8).reshape(height, weight)

#写入原始图像数据
def write_raw(file_name, array):
    info = array.shape
    weight = info[1]
    height = info[0]
    with open(file_name, 'wb') as f:
        f.write(struct.pack("II", weight, height))
        for row in array:
            for elem in row:
                f.write(struct.pack("B", elem))

#读取并重新保存原始图像
def read_and_save_raw_images():
    global raw_files,dir_path,current_index
    current_index = 0
    dir_path = filedialog.askdirectory()
    new_path = os.path.join(dir_path, "new")
    raw_files = []
    if dir_path:
        for raw_file in os.listdir(dir_path):
            if raw_file.endswith(".raw"):
                raw_files.append(raw_file)
    raw_files = sorted(raw_files,key=extract_parts)#对文件名进行排序
    displayed = False
    for raw_file in raw_files:
        image_array = read_raw(os.path.join(dir_path, raw_file))
        #默认显示第一张图片
        if not displayed:
            display_image(image_array,file_name=raw_file, is_color=(len(image_array.shape)) == 3)
            displayed = True
        os.makedirs(new_path, exist_ok=True)
        new_file = os.path.join(new_path, f"new_{raw_file}")
        write_raw(new_file, image_array)
        if np.array_equal(read_raw(new_file), image_array):
            print(f"{raw_file}副本创建完毕，经检查，内容格式一致")
        else:
            print(f"{raw_file}副本创建完毕，经检查，内容格式不一致")
    print(f"共读取{len(raw_files)}个RAW文件，已重新保存至路径{new_path}")
    show_tips()



#显示下一张图片
def img_forward(event):
    global current_index, raw_files, dir_path
    if len(raw_files) > 0:
        current_index = (current_index + 1) % len(raw_files)  # 循环显示
        image_array = read_raw(os.path.join(dir_path, raw_files[current_index]))
        display_image(image_array,file_name=raw_files[current_index])

#显示上一张图片
def img_backward(event):
    global current_index, raw_files, dir_path
    if len(raw_files) > 0:
        current_index = (current_index - 1) % len(raw_files)  # 循环显示
        image_array = read_raw(os.path.join(dir_path, raw_files[current_index]))
        display_image(image_array, file_name=raw_files[current_index])



#生成不同类型的图像
def generate_black_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    display_image(image_array,file_name="全黑图像")
    clear_tips()

def generate_white_image():
    image_array = np.full((512, 512), 255, dtype=np.uint8)
    display_image(image_array,file_name="全白图像")
    clear_tips()

def generate_gradient_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    for x in range(512):
        value = min(x, 255)
        image_array[:, x] = value
    display_image(image_array,file_name="灰度渐变图像")
    clear_tips()

def generate_color_image():
    image_array = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        for y in range(512):
            red = min(0.5 * x, 255)
            green = min(0.5 * y, 255)
            blue = min((1 * x + 9 * y) / 10, 255)
            image_array[y, x] = [red, green, blue]
    display_image(image_array, is_color=True,file_name="彩色图像")
    clear_tips()

def generate_stripes_image():
    image_array = np.zeros((512, 512), dtype=np.uint8)
    values = [0, 31, 63, 95, 127, 159, 191, 224, 255]
    width = 512 // len(values)
    for i in range(1, len(values) + 1):
        value = values[i - 1]
        start = i * width
        end = (i + 1) * width
        image_array[:, start:end] = value
    display_image(image_array,file_name="条纹图像")
    clear_tips()

#清除提示
def clear_tips():
    tips_forward.config(text="")
    tips_backward.config(text="")
#显示提示
def show_tips():
    tips_forward.config(text="左击查看下一张raw图像")
    tips_backward.config(text="右击查看上一张raw图像")

if __name__ == '__main__':

    gui = tk.Tk()
    gui.title("数字图像处理实验二 图像文件读写")

    label = Label(gui, text="图像展示", font=("楷体", 16))
    label.grid(row=0, column=0, padx=10, pady=10)

    canvas = Canvas(gui, width=width_canvas, height=height_canvas, background=rgb_to_hex(80, 140, 100))
    canvas.grid(row=1, column=0, padx=20, pady=10)

    #信息显示区
    info_frame = Frame(gui)
    info_frame.grid(row=2, column=0, padx=20, pady=10)


    info_name = Label(info_frame, text="名称：", font=("楷体", 14))
    info_name.grid(row=0, column=0, padx=0, pady=0)

    info_color = Label(info_frame, text="通道：", font=("楷体", 14))
    info_color.grid(row=1, column=0, padx=10, pady=0)

    info_weight = Label(info_frame, text="高：", font=("楷体", 14))
    info_weight.grid(row=2, column=0, padx=10, pady=0)

    info_height = Label(info_frame, text="宽：", font=("楷体", 14))
    info_height.grid(row=3, column=0, padx=10, pady=0)

    #提示区
    tips_frame = Frame(gui)
    tips_frame.grid(row=2, column=1, padx=20, pady=10)

    tips_forward = Label(tips_frame, text="", font=("楷体", 14), fg="blue")
    tips_forward.grid(row=0, column=0, padx=0, pady=3)
    tips_backward = Label(tips_frame, text="", font=("楷体", 14), fg="blue")
    tips_backward.grid(row=1, column=0, padx=0, pady=3)

    raw_files=[]
    # 绑定鼠标左键和右键
    canvas.bind("<Button-1>", img_forward)  # 左键：下一张
    canvas.bind("<Button-3>", img_backward)  # 右键：上一张


    # 按钮框架在右侧
    button_frame = Frame(gui,bg="lightgray")
    button_frame.grid(row=1, column=1, padx=20, pady=10,)

    buttons = [
        ("全黑图像", generate_black_image),
        ("全白图像", generate_white_image),
        ("渐变图像", generate_gradient_image),
        ("彩色图像", generate_color_image),
        ("条纹图像", generate_stripes_image),
        ("打开文件", open_image),
        ("打开RAW图像", read_and_save_raw_images)
    ]

    for i, (text, command) in enumerate(buttons):
        button = Button(button_frame, text=text, command=command, font=("楷体", 18))
        button.grid(row=i, column=0, padx=15, pady=10)

    gui.mainloop()
