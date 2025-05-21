import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import numpy as np
import struct
import cv2




class ImageFilter():
    # 滑动条列表
    scales = []
    def __init__(self, gui):
        self.gui = gui
        self.gui.title("数字图像处理实验七 滤波器设计")

        # 顶部按钮框架
        self.frame = ttk.Frame(gui)
        self.frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=0)

        # 按钮控件
        self.btn_load = ttk.Button(self.frame, text="加载图像", command=self.open_image)
        self.btn_load.grid(row=0, column=0, padx=5, pady=5)

        # 功能选择标签
        self.label_choose = ttk.Label(self.frame, text="操作选择")
        self.label_choose.grid(row=0, column=1, padx=5, pady=5)
        self.label_choose.config(state=tk.DISABLED)

        # 功能选择下拉菜单
        self.mode_menu = ttk.Combobox(self.frame, values=["均值滤波", "中值滤波", "高斯滤波", "Sobel滤波", "Laplace滤波"])
        self.mode_menu.grid(row=0, column=2, padx=5, pady=5)
        self.mode_menu.config(state=tk.DISABLED)

        # 将选择事件绑定到对应的函数
        self.mode_menu.bind("<<ComboboxSelected>>", self.on_mode_changed)

        # 创建一个变量，用于表示是否启用与cv2库函数的对比
        self.cv2_check = tk.IntVar()

        # 添加与cv2对比的勾选框
        self.cv2_check_button = ttk.Checkbutton(self.frame, text="与库函数进行对比", variable=self.cv2_check, command=self.on_check)
        self.cv2_check_button.grid(row=0, column=6, padx=5, pady=5, sticky="ew")

        # 均值滤波卷积核大小
        self.mean_filter_ksize_scale = Scale(self.frame, from_=3, to=35, length=200, orient=tk.HORIZONTAL,
                                             label="卷积核大小", command=self.on_scale_changed_mean_filter)
        self.mean_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
        self.mean_filter_ksize_scale.grid_forget()

        # 中值滤波卷积核大小
        self.median_filter_ksize_scale = Scale(self.frame, from_=3, to=35, length=200, orient=tk.HORIZONTAL,
                                             label="卷积核大小", command=self.on_scale_changed_median_filter, resolution=2)
        self.median_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
        self.median_filter_ksize_scale.grid_forget()

        # 高斯滤波卷积核大小
        self.gaussian_filter_ksize_scale = Scale(self.frame, from_=3, to=35, length=200, orient=tk.HORIZONTAL,
                                               label="卷积核大小", command=self.on_scale_changed_gaussian_filter, resolution=2)
        self.gaussian_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
        self.gaussian_filter_ksize_scale.grid_forget()

        # 高斯函数标准差大小
        self.gaussian_filter_sigma_scale = Scale(self.frame, from_=1, to=10, length=100, orient=tk.HORIZONTAL,
                                               label="高斯函数标准差大小", command=self.on_scale_changed_gaussian_filter)
        self.gaussian_filter_sigma_scale.grid(row=0, column=3, padx=5, pady=5)
        self.gaussian_filter_sigma_scale.grid_forget()

        # Sobel方向选择标签
        self.label_sobel = ttk.Label(self.frame, text="检测方向")
        self.label_sobel.grid(row=0, column=3, padx=5, pady=5)
        self.label_sobel.config(state=tk.DISABLED)
        self.label_sobel.grid_forget()

        # Sobel方向选择下拉菜单
        self.dir_sobel_menu = ttk.Combobox(self.frame, values=["水平检测", "竖直检测", "组合检测"])
        self.dir_sobel_menu.grid(row=0, column=4, padx=5, pady=5)
        self.dir_sobel_menu.set("水平检测")
        self.dir_sobel_menu.grid_forget()

        # 将选择事件绑定到对应的函数
        self.dir_sobel_menu.bind("<<ComboboxSelected>>", self.on_sobel_menu_changed)

        self.scales.append(self.mean_filter_ksize_scale)
        self.scales.append(self.median_filter_ksize_scale)
        self.scales.append(self.gaussian_filter_ksize_scale)
        self.scales.append(self.gaussian_filter_sigma_scale)
        self.scales.append(self.label_sobel)
        self.scales.append(self.dir_sobel_menu)

        # 画布（用于显示原始图像）
        self.canvas = tk.Canvas(gui, bg="gray")
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        self.label_canvas = tk.Label(gui, text="原始图像")
        self.label_canvas.grid(row=2, column=0, padx=10, pady=10)

        # 画布（用于显示处理后的图像）
        self.canvas_new = tk.Canvas(gui, bg="gray")
        self.canvas_new.grid(row=1, column=1, padx=10, pady=10)
        self.label_canvas_new = tk.Label(gui, text="处理后的图像")
        self.label_canvas_new.grid(row=2, column=1, padx=10, pady=10)

        # 画布（用于显示库函数处理后的图像）
        self.canvas_cv = tk.Canvas(gui, bg="gray")
        self.canvas_cv.grid(row=1, column=2, padx=10, pady=10)
        self.canvas_cv.grid_forget()
        self.label_canvas_cv = tk.Label(gui, text="库函数处理后的图像")
        self.label_canvas_cv.grid(row=2, column=2, padx=10, pady=10)
        self.label_canvas_cv.grid_forget()

        # 配置 grid 行列权重，使画布可以自动扩展
        self.gui.grid_rowconfigure(1, weight=1)
        self.gui.grid_columnconfigure(0, weight=1)

    # 原始图像
    image_array_original = None
    image_original = None
    # 处理后的图像
    image_array_proc = None
    image_proc = None
    # 库函数处理后的图像
    image_array_cv2 = None
    image_cv2 = None

    # 当模式选择下拉菜单改动时触发
    def on_mode_changed(self, _):
        # 获取当前选择的模式
        selected_value = self.mode_menu.get()
        # 隐藏所有滑动条
        for scale in self.scales:
            scale.grid_forget()
        if selected_value == "均值滤波":
            self.mean_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
            self.on_scale_changed_mean_filter(None)
        elif selected_value == "中值滤波":
            self.median_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
            self.on_scale_changed_median_filter(None)
        elif selected_value == "高斯滤波":
            self.gaussian_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
            self.gaussian_filter_sigma_scale.grid(row=0, column=4, padx=5, pady=5)
            self.on_scale_changed_gaussian_filter(None)
        elif selected_value == "Sobel滤波":
            self.label_sobel.grid(row=0, column=3, padx=5, pady=5)
            self.dir_sobel_menu.grid(row=0, column=4, padx=5, pady=5)
            self.on_sobel_menu_changed("")
        elif selected_value == "Laplace滤波":
            self.img_Laplace_filter(self.image_array_original)
            if self.cv2_check.get():
                self.image_array_cv2 = cv2.Laplacian(self.image_array_original, cv2.CV_64F, ksize=1)
                self.display_image(2)

    # 当均值滤波模式下的滑动条改动时触发
    def on_scale_changed_mean_filter(self, _):
        # 获取改动后的值
        selected_value = int(self.mean_filter_ksize_scale.get())
        # 调用均值滤波函数
        self.img_mean_filter(selected_value)
        # 显示处理后的图像
        self.display_image(1)
        if self.cv2_check.get():
            self.image_array_cv2 = cv2.blur(self.image_array_original, (selected_value, selected_value))
            self.display_image(2)

    # 当均中值波模式下的滑动条改动时触发
    def on_scale_changed_median_filter(self, _):
        # 获取改动后的值
        selected_value = int(self.median_filter_ksize_scale.get())
        # 调用中值滤波函数
        self.img_median_filter(selected_value)
        # 显示处理后的图像
        self.display_image(1)
        if self.cv2_check.get():
            self.image_array_cv2 = cv2.medianBlur(self.image_array_original, selected_value)
            self.display_image(2)

    def on_scale_changed_gaussian_filter(self, _):
        # 获取改动后的值
        ksize = int(self.gaussian_filter_ksize_scale.get())
        sigma = int(self.gaussian_filter_sigma_scale.get())
        # 调用高斯滤波函数
        self.img_Gaussian_filter(self.image_array_original, ksize, sigma)
        # 显示处理后的图像
        self.display_image(1)
        if self.cv2_check.get():
            self.image_array_cv2 = cv2.GaussianBlur(self.image_array_original, (ksize, ksize), sigma)
            self.display_image(2)

    def on_sobel_menu_changed(self, _):
        selected_value = self.dir_sobel_menu.get()
        if selected_value == "水平检测":
            self.img_Sobel_filter(self.image_array_original, "x")
        elif selected_value == "竖直检测":
            self.img_Sobel_filter(self.image_array_original, "y")
        elif selected_value == "组合检测":
            self.img_Sobel_filter(self.image_array_original, "xy")
        self.display_image(1)
        if self.cv2_check.get():
            grad_x = cv2.Sobel(self.image_array_original, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(self.image_array_original, cv2.CV_64F, 0, 1, ksize=3)
            if selected_value == '水平检测':
                self.image_array_cv2 = grad_x
            elif selected_value == '竖直检测':
                self.image_array_cv2 = grad_y
            elif selected_value == '组合检测':
                self.image_array_cv2 = cv2.magnitude(grad_x, grad_y)
            self.display_image(2)

    # 当启用/禁用库函数对比
    def on_check(self):
        if self.cv2_check.get():
            self.canvas_cv.grid(row=1, column=2, padx=10, pady=10)
            self.label_canvas_cv.grid(row=2, column=2, padx=10, pady=10)
        else:
            self.canvas_cv.grid_forget()
            self.label_canvas_cv.grid_forget()

    # 打开图像
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.raw;*.jpg;*.png;*.bmp;*.jpeg")])
        if not file_path:
            return
        try:
            # 判断图片类型
            if file_path.endswith(".raw"):
                with open(file_path, 'rb') as f:
                    weight, height = struct.unpack("II", f.read(8))
                    data_size = weight * height
                    raw_data = struct.unpack(f"{data_size}B", f.read(data_size))
                self.image_array_original = np.array(raw_data, np.uint8).reshape(height, weight)
            else:
                self.image_array_original = np.array(Image.open(file_path), dtype=np.uint8)
                if self.image_array_original.ndim > 2:
                    messagebox.showerror("错误", "请打开.raw文件或单通道灰度图")
                    return
            # 清除上一张原始图像
            self.canvas.delete("all")
            # 清除上一次操作结果
            self.canvas_new.delete("all")
            # 清除库函数处理结果
            self.canvas_cv.delete("all")
            self.display_image(0)
            self.image_array_proc = None
            self.display_image(1)
            self.image_array_cv2 = None
            self.display_image(2)
            # 启用操作选择栏
            self.mode_menu.config(state=tk.NORMAL)
            # 启用操作选择栏标签
            self.label_choose.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图像: {str(e)}")

    def display_image(self, place):
        # 显示原始图像
        if place == 0:
            if self.image_array_original is not None:
                pil_image = Image.fromarray(self.image_array_original)
                width, height = pil_image.size
                self.image_original = ImageTk.PhotoImage(pil_image)
                # 在画布上显示图片
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_original)
                # 调整画布自适应图片大小
                self.canvas.config(height=height, width=width)
        # 显示分割后的图像
        elif place == 1:
            if self.image_array_proc is not None:
                pil_image = Image.fromarray(self.image_array_proc)
                width, height = pil_image.size
                self.image_proc = ImageTk.PhotoImage(pil_image)
                self.canvas_new.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
                self.canvas_new.config(height=height, width=width)
        elif place == 2:
            if self.image_array_cv2 is not None:
                pil_image = Image.fromarray(self.image_array_cv2)
                width, height = pil_image.size
                self.image_cv2 = ImageTk.PhotoImage(pil_image)
                self.canvas_cv.create_image(0, 0, anchor=tk.NW, image=self.image_cv2)
                self.canvas_cv.config(height=height, width=width)

    # 二维卷积函数(接收参数image为待处理图像，kernel为卷积核)
    @staticmethod
    def convolve2d(image, kernel):
        # 获取图片宽高
        img_h, img_w = image.shape
        # 获取卷积核宽高
        k_h, k_w = kernel.shape
        # 获取边缘填充的大小（至少为卷积核大小的一半）
        pad_h, pad_w = k_h // 2, k_w // 2
        # 边缘填充
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        # 生成与原图片大小一至的空图像，用于存放处理后的结果
        result = np.zeros((img_h, img_w), dtype=np.float32)
        # result = np.zeros_like(image)
        # 卷积计算
        for i in range(img_h):
            for j in range(img_w):
                # 取出与卷积核大小相同的区域
                region = padded_img[i:i + k_h, j:j + k_w]
                # 区域与卷积核所有元素相乘并求和
                result[i, j] = np.sum(region * kernel)
        # # 确保灰度值在0~255，并返回结果
        # return np.clip(result, 0, 255).astype(np.uint8)
        return result

    # 均值滤波器
    def img_mean_filter(self, ksize=3):
        # 均值滤波器卷积核并归一化
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        # 调用二维卷积运算并返回结果
        self.image_array_proc = self.convolve2d(self.image_array_original, kernel)

    # 中值滤波器
    def img_median_filter(self, ksize=3):
        # 获取原始图片的宽高
        img_h, img_w = self.image_array_original.shape
        # 获取边缘填充的大小（至少为卷积核大小的一半）
        pad = ksize // 2
        # 边缘填充
        padded_img = np.pad(self.image_array_original, ((pad, pad), (pad, pad)), mode='reflect')
        # 生成与原图片大小一至的空图像，用于存放处理后的结果
        result = np.zeros_like(self.image_array_original)
        # 中值滤波
        for i in range(img_h):
            for j in range(img_w):
                # 取出与核大小相同的区域
                region = padded_img[i:i + ksize, j:j + ksize]
                # 取中值作为当前像素点灰度值
                result[i, j] = np.median(region)
        # 返回结果
        self.image_array_proc = result

    # 生成高斯卷积核
    @staticmethod
    def gaussian_kernel(size, sigma=1):
        # 生成 从 -kernel_size//2 到 kernel_size//2 的等距坐标点
        ax = np.linspace(-(size // 2), size // 2, size)
        # 生成网格坐标矩阵，用来计算二维高斯函数的值
        xx, yy = np.meshgrid(ax, ax)
        # 计算高斯函数
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        # 归一化卷积核
        return kernel / np.sum(kernel)

    # 高斯滤波器
    def img_Gaussian_filter(self, image, ksize=3, sigma=1):
        kernel = self.gaussian_kernel(ksize, sigma)
        self.image_array_proc = self.convolve2d(image, kernel)

    # Sobel滤波器-边缘检测
    def img_Sobel_filter(self, image, mode):
        grad = None
        # 水平方向检测核
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        # 竖直方向检测核
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=np.float32)
        # 判断模式
        if mode.lower() == "x":
            # 利用x方向的梯度进行计算
            grad = self.convolve2d(image, sobel_x)
        elif mode.lower() == "y":
            # 利用y方向的梯度进行计算
            grad = self.convolve2d(image, sobel_y)
        elif mode.lower() == "xy":
            # 利用x方向的梯度进行计算
            gx = self.convolve2d(image, sobel_x)
            # 利用y方向的梯度进行计算
            gy = self.convolve2d(image, sobel_y)
            # 组合两个方向的结果
            grad = np.hypot(gx, gy)
        self.image_array_proc = np.clip(grad, 0, 255).astype(np.uint8)

    # Laplace滤波器
    def img_Laplace_filter(self, image):
        # 定义拉普拉斯卷积核
        laplace_kernel = np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=np.float32)
        self.image_array_proc = self.convolve2d(image, laplace_kernel)
        # 显示图片
        self.display_image(1)


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageFilter(root)
    root.mainloop()
