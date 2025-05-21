import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Label, Button, Frame, Scale
from PIL import Image, ImageTk
import numpy as np
import struct
import math


class ImageTransformer:
    def __init__(self, gui):
        self.gui = gui
        self.gui.title("数字图像处理实验五 几何变换")

        self.label_original_image = Label(gui, text="原始图像", font=("楷体", 16))
        self.label_original_image.grid(row=0, column=0, padx=20, pady=10)

        self.label_image = Label(gui, text="变换后的图像", font=("楷体", 16))
        self.label_image.grid(row=0, column=1, padx=20, pady=10)

        self.canvas_original = Canvas(root)
        self.canvas_original.grid(row=1, column=0, padx=20, pady=10)
        
        self.canvas = Canvas(root)
        self.canvas.grid(row=1, column=1, padx=20, pady=10)

        self.btn_frame = Frame(root)
        self.btn_frame.grid(row=2, column=0, padx=20, pady=10)

        btn_open = Button(self.btn_frame, text="打开图像", width=8, height=1, font=("楷体", 16), command=self.open_image)
        btn_open.grid(row=0, column=0, padx=20, pady=10)

        btn_rl_90 = Button(self.btn_frame, text="逆时针旋转90°", font=("楷体", 16), command=self.image_transpose)
        btn_rl_90.grid(row=1, column=0, padx=20, pady=10)

        btn_rl_10 = Button(self.btn_frame, text="逆时针旋转10°", font=("楷体", 16), command=lambda: self.image_rotate(-10))
        btn_rl_10.grid(row=1, column=1, padx=20, pady=10)

        btn_ex = Button(self.btn_frame, text="放大为原来的2倍", font=("楷体", 16), command=self.image_zoom_in)
        btn_ex.grid(row=2, column=0, padx=20, pady=10)

        btn_na = Button(self.btn_frame, text="缩小为原来的1/2", font=("楷体", 16), command=self.image_zoom_out)
        btn_na.grid(row=2, column=1, padx=20, pady=10)

        scale_frame = Frame(gui)
        scale_frame.grid(row=2, column=1, padx=20, pady=10)

        self.dx_scale = Scale(scale_frame, from_=-500, to=500, length=400, orient=tk.HORIZONTAL, label="水平位移",
                              command=self.update_image_translate)
        self.dx_scale.set(0)
        self.dx_scale.grid(row=0, column=0, padx=20, pady=10)

        self.dy_scale = Scale(scale_frame, from_=-500, to=500, length=400, orient=tk.HORIZONTAL, label="垂直位移",
                              command=self.update_image_translate)
        self.dy_scale.set(0)
        self.dy_scale.grid(row=1, column=0, padx=20, pady=10)

        # 图像变量
        self.image_array_original = None
        self.tk_image_original = None

        self.image_array_new = None
        self.tk_image_new = None

    image_array_original = None

    # 打开图像（支持通用格式以及raw文件）
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.raw;*.jpeg")])
        if not file_path:
            return
        try:
            self.image_array_original = np.array(Image.open(file_path), dtype=np.uint8)
            self.display_image()
            self.image_array_new = None
            self.display_image(False)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图像: {str(e)}")

    # 静态方法
    @staticmethod
    def open_raw(path):
        with open(path, 'rb') as f:
            weight, height = struct.unpack("II", f.read(8))
            data_size = weight * height
            raw_data = struct.unpack(f"{data_size}B", f.read(data_size))
        return np.array(raw_data, np.uint8).reshape(height, weight)

    def display_image(self, is_original=True):
        if is_original:
            if self.image_array_original is not None:
                pil_image = Image.fromarray(self.image_array_original)
                width, height = pil_image.size
                self.tk_image_original = ImageTk.PhotoImage(pil_image)
                self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.tk_image_original)
                self.canvas_original.config(height=height, width=width)
        else:
            if self.image_array_new is not None:
                pil_image = Image.fromarray(self.image_array_new)
                width, height = pil_image.size
                self.tk_image_new = ImageTk.PhotoImage(pil_image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_new)
                self.canvas.config(height=height, width=width)
            else:
                # 清除上一次显示的结果
                self.canvas.delete("all")

    def image_transpose(self):
        self.image_rotate(-90)

    def image_rotate(self, angle):
        if self.image_array_original is not None:
            height, width = self.image_array_original.shape[:2]

            # 将角度转换为弧度
            angle_rad = math.radians(angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            # 计算旋转后图像的新尺寸
            # 四个角在旋转后的坐标位置
            corners = [
                (0, 0),
                (width, 0),
                (0, height),
                (width, height)
            ]

            # 旋转四个角以计算边界框
            new_corners = []
            for x, y in corners:
                new_x = x * cos_angle - y * sin_angle
                new_y = x * sin_angle + y * cos_angle
                new_corners.append((new_x, new_y))

            # 计算旋转后图像的边界大小
            x_coords, y_coords = zip(*new_corners)
            new_width = int(max(x_coords) - min(x_coords))
            new_height = int(max(y_coords) - min(y_coords))

            # 创建新的图像并填充黑色
            if len(self.image_array_original.shape) == 3:
                new_image = np.zeros((new_height, new_width, self.image_array_original.shape[2]), dtype=np.uint8)
            else:
                new_image = np.zeros((new_height, new_width), dtype=np.uint8)

            # 新图像的中心点
            center_x, center_y = new_width // 2, new_height // 2

            # 源图像的中心点
            src_center_x, src_center_y = width // 2, height // 2

            # 遍历目标图像每个像素
            for y in range(new_height):
                for x in range(new_width):
                    # 将目标像素映射回原图像（反向映射）
                    # 计算相对于新图中心的偏移量
                    dx = x - center_x
                    dy = y - center_y

                    # 映射回原图
                    old_x = dx * cos_angle + dy * sin_angle + src_center_x
                    old_y = -dx * sin_angle + dy * cos_angle + src_center_y

                    # 判断是否在原图范围内
                    if 0 <= old_x < width and 0 <= old_y < height:
                        # 使用双线性插值计算源像素的值
                        new_image[y, x] = self.bilinear_interpolation(old_x, old_y)

            # 显示旋转后的图像
            self.image_array_new = new_image
            self.display_image(False)

    def image_translate(self, dx=0, dy=0):
        if self.image_array_original is not None:
            # 获取图像大小
            height, width = self.image_array_original.shape[:2]
            # 创建目标图像
            if len(self.image_array_original.shape) == 3:  # 彩色图像
                self.image_array_new = np.zeros((height, width, self.image_array_original.shape[2]), dtype=np.uint8)
            else:  # 灰度图像
                self.image_array_new = np.zeros((height, width), dtype=np.uint8)
            # 遍历原图像像素
            for y in range(height):
                for x in range(width):
                    # 计算新的位置
                    new_x = min(width - 1, max(0, x + dx))
                    new_y = min(height - 1, max(0, y + dy))

                    self.image_array_new[new_y, new_x] = self.image_array_original[y, x]
            self.tk_image_new = Image.fromarray(self.image_array_new)

    def update_image_translate(self, _):
        if self.image_array_original is not None:
            dx = int(self.dx_scale.get())
            dy = int(self.dy_scale.get())
            self.image_translate(dx=dx, dy=dy)
            self.display_image(False)

    # 双线性插值算法
    def bilinear_interpolation(self, x, y):
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, self.image_array_original.shape[1] - 1)
        y1 = min(y0 + 1, self.image_array_original.shape[0] - 1)

        # 获取四个邻域像素的值
        q11 = self.image_array_original[y0, x0]
        q21 = self.image_array_original[y0, x1]
        q12 = self.image_array_original[y1, x0]
        q22 = self.image_array_original[y1, x1]

        # 计算水平插值
        r1 = (x1 - x) * q11 + (x - x0) * q21
        r2 = (x1 - x) * q12 + (x - x0) * q22

        # 计算垂直插值
        p = (y1 - y) * r1 + (y - y0) * r2

        return p

    # 将图像放大为原来的2倍
    def image_zoom_in(self):
        if self.image_array_original is not None:
            height, width = self.image_array_original.shape[:2]
            new_height, new_width = height * 2, width * 2

            if len(self.image_array_original.shape) == 3:  # 彩色图像
                new_image = np.zeros((new_height, new_width, self.image_array_original.shape[2]), dtype=np.uint8)
            else:  # 灰度图像
                new_image = np.zeros((new_height, new_width), dtype=np.uint8)

            for y in range(new_height):
                for x in range(new_width):
                    # 使用双线性插值来计算目标位置的像素值
                    # 将目标图像的像素位置映射到源图像的浮动位置
                    source_x = x / 2
                    source_y = y / 2
                    new_image[y, x] = self.bilinear_interpolation(source_x, source_y)

            self.image_array_new = new_image
            self.display_image(False)  # 更新图像显示

    # 将图像缩小为原来的1/2
    def image_zoom_out(self):
        if self.image_array_original is not None:
            height, width = self.image_array_original.shape[:2]
            # 除二并取整
            new_height, new_width = height // 2, width // 2

            if len(self.image_array_original.shape) == 3:  # 彩色图像
                new_image = np.zeros((new_height, new_width, self.image_array_original.shape[2]), dtype=np.uint8)
            else:  # 灰度图像
                new_image = np.zeros((new_height, new_width), dtype=np.uint8)

            for y in range(new_height):
                for x in range(new_width):
                    # 使用双线性插值来计算目标位置的像素值
                    source_x = x * 2
                    source_y = y * 2
                    new_image[y, x] = self.bilinear_interpolation(source_x, source_y)

            self.image_array_new = new_image
            self.display_image(False)  # 更新图像显示


# 启动GUI
root = tk.Tk()
app = ImageTransformer(root)
root.mainloop()
