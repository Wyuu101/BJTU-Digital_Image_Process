import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import numpy as np
import struct
from numpy.lib.stride_tricks import as_strided


class ImageFilter():
    # 滑动条列表
    scales = []
    def __init__(self, gui):
        self.gui = gui
        self.gui.title("数字图像处理实验八 基于边缘的图像分割")

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
        self.mode_menu = ttk.Combobox(self.frame, values=["Sobel边缘检测", "Canny边缘检测"])
        self.mode_menu.grid(row=0, column=2, padx=5, pady=5)
        self.mode_menu.config(state=tk.DISABLED)

        # 将选择事件绑定到对应的函数
        self.mode_menu.bind("<<ComboboxSelected>>", self.on_mode_changed)

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
        self.scales.append(self.label_sobel)
        self.scales.append(self.dir_sobel_menu)

        # Canny边缘提取-高斯滤波卷积核大小
        self.gaussian_filter_ksize_scale = Scale(self.frame, from_=3, to=10, length=100, orient=tk.HORIZONTAL,
                                                 label="高斯卷积核大小", command=self.on_scale_changed_canny,
                                                 resolution=2)
        self.gaussian_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
        self.gaussian_filter_ksize_scale.grid_forget()

        # Canny边缘提取-高斯滤波sigma大小
        self.gaussian_filter_sigma_scale = Scale(self.frame, from_=1, to=10, length=100, orient=tk.HORIZONTAL,
                                                 label="高斯滤波sigma值", command=self.on_scale_changed_canny,
                                                 resolution=1)
        self.gaussian_filter_sigma_scale.grid(row=0, column=4, padx=5, pady=5)
        self.gaussian_filter_sigma_scale.grid_forget()


        # Canny边缘提取-上阈值
        self.canny_threshold_up = Scale(self.frame, from_=0, to=255, length=200, orient=tk.HORIZONTAL,
                                                 label="边缘跟踪-上阈值", command=self.on_scale_changed_canny,
                                                 resolution=1)
        self.canny_threshold_up.grid(row=0, column=6, padx=5, pady=5)
        self.canny_threshold_up.set(100)
        self.canny_threshold_up.grid_forget()

        # Canny边缘提取-下阈值
        self.canny_threshold_down = Scale(self.frame, from_=0, to=255, length=200, orient=tk.HORIZONTAL,
                                                 label="边缘跟踪-下阈值", command=self.on_scale_changed_canny,
                                                 resolution=1)
        self.canny_threshold_down.grid(row=0, column=5, padx=5, pady=5)
        self.canny_threshold_down.set(50)
        self.canny_threshold_down.grid_forget()

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

        # 配置 grid 行列权重，使画布可以自动扩展
        self.gui.grid_rowconfigure(1, weight=1)
        self.gui.grid_columnconfigure(0, weight=1)

    # 原始图像
    image_array_original = None
    image_original = None
    # 处理后的图像
    image_array_proc = None
    image_proc = None


    # 当模式选择下拉菜单改动时触发
    def on_mode_changed(self, _):
        # 获取当前选择的模式
        selected_value = self.mode_menu.get()
        # 隐藏所有滑动条
        for scale in self.scales:
            scale.grid_forget()
        if selected_value == "Sobel边缘检测":
            self.label_sobel.grid(row=0, column=3, padx=5, pady=5)
            self.dir_sobel_menu.grid(row=0, column=4, padx=5, pady=5)
            self.on_sobel_menu_changed("")
        elif selected_value == "Canny边缘检测":
            self.gaussian_filter_ksize_scale.grid(row=0, column=3, padx=5, pady=5)
            self.gaussian_filter_sigma_scale.grid(row=0, column=4, padx=5, pady=5)
            self.canny_threshold_up.grid(row=0, column=6, padx=5, pady=5)
            self.canny_threshold_down.grid(row=0, column=5, padx=5, pady=5)
            self.on_scale_changed_canny(self)
            self.display_image(1)

    # 当sobel检测方向选项改变时触发
    def on_sobel_menu_changed(self, _):
        selected_value = self.dir_sobel_menu.get()
        if selected_value == "水平检测":
            self.Sobel_segmentation(self.image_array_original, "x")
        elif selected_value == "竖直检测":
            self.Sobel_segmentation(self.image_array_original, "y")
        elif selected_value == "组合检测":
            self.Sobel_segmentation(self.image_array_original, "xy")
        self.display_image(1)

    def on_scale_changed_canny(self, _):
        # 获取改动后的值
        ksize = int(self.gaussian_filter_ksize_scale.get())
        sigma = int(self.gaussian_filter_sigma_scale.get())
        canny_threshold_up = int(self.canny_threshold_up.get())
        canny_threshold_down = int(self.canny_threshold_down.get())
        # 调用高斯滤波函数
        self.Canny_segmentation(self.image_array_original, ksize, sigma, canny_threshold_up, canny_threshold_down)
        # 显示处理后的图像
        self.display_image(1)

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
            self.display_image(0)
            self.image_array_proc = None
            self.display_image(1)
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

    # 二维卷积函数(接收参数image为待处理图像，kernel为卷积核)
    @staticmethod
    def convolve2d(image, kernel):
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # 填充图像
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # 获取滑动窗口视图
        shape = (img_h, img_w, k_h, k_w)
        strides = (padded_img.strides[0], padded_img.strides[1], padded_img.strides[0], padded_img.strides[1])
        windows = as_strided(padded_img, shape=shape, strides=strides)

        # 执行卷积（批量点乘后求和）
        result = np.einsum('ijkl,kl->ij', windows, kernel).astype(np.float32)

        return result

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
    def Sobel_segmentation(self, image, mode):
        gx = None
        gy = None
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
        if mode.lower() == "xy":
            return gx, gy, grad

    def Canny_segmentation(self, image, ksize=3, sigma=1, low_threshold=50, high_threshold=100):
        # 先进行高斯滤波
        self.img_Gaussian_filter(image, ksize, sigma)
        # 进行Sobel组合检测，并得到梯度幅度矩阵和方向矩阵
        gx, gy, grad = self.Sobel_segmentation(self.image_array_proc, "xy")
        direction = np.arctan2(gy, gx)
        # 非极大值检测
        nms = self.non_maximum_suppression(grad, direction)
        # 边缘跟踪
        edges = self.hysteresis_threshold(nms, low_threshold, high_threshold)
        self.image_array_proc = (edges * 255).astype(np.uint8)

    def non_maximum_suppression(self, magnitude, direction):
        # 获取图像尺寸（高 h，宽 w）
        h, w = magnitude.shape
        # 创建结果矩阵，初始为全 0，类型为 float32
        result = np.zeros((h, w), dtype=np.float32)
        # 将梯度方向（弧度）转换为角度（0~180度）
        angle = direction * 180. / np.pi
        # 将角度中小于 0 的部分调整为正值（统一范围）
        angle[angle < 0] += 180
        # 初始化一个与角度数组大小相同的矩阵，用于存储方向分类结果
        angle_bin = np.zeros_like(angle, dtype=np.uint8)
        # 将角度归类为4个主方向（0°, 45°, 90°, 135°）
        angle_bin[((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))] = 0
        angle_bin[(22.5 <= angle) & (angle < 67.5)] = 45
        angle_bin[(67.5 <= angle) & (angle < 112.5)] = 90
        angle_bin[(112.5 <= angle) & (angle < 157.5)] = 135
        # 定义每个主方向对应的两个对比像素点的坐标偏移值（dy, dx）
        offsets = {
            0: ((0, 1), (0, -1)),  # 水平方向
            45: ((1, -1), (-1, 1)),  # 右下-左上方向
            90: ((1, 0), (-1, 0)),  # 垂直方向
            135: ((-1, -1), (1, 1)),  # 左上-右下方向
        }
        # 遍历每一个方向进行非极大值抑制
        for angle_val in [0, 45, 90, 135]:
            # 获取当前方向的掩码区域
            mask = (angle_bin == angle_val)
            # 取出当前中心像素区域（裁去边缘1个像素）
            (dy1, dx1), (dy2, dx2) = offsets[angle_val]
            mag_center = magnitude[1:-1, 1:-1]
            # 获取方向上前后两个邻域像素值
            mag_q = magnitude[1 + dy1:h - 1 + dy1, 1 + dx1:w - 1 + dx1]
            mag_r = magnitude[1 + dy2:h - 1 + dy2, 1 + dx2:w - 1 + dx2]
            # 同样裁剪掩码区域，使其与中心区域对齐
            mask_crop = mask[1:-1, 1:-1]
            # 若中心像素大于两个方向像素且属于当前方向，则保留
            cond = (mag_center >= mag_q) & (mag_center >= mag_r) & mask_crop
            result[1:-1, 1:-1][cond] = mag_center[cond]
        # 返回抑制后的结果图像
        return result

    def hysteresis_threshold(self, img, low, high):
        # 识别出强边缘：像素值大于高阈值的为强边缘
        strong = (img > high)
        # 识别出弱边缘：像素值介于低阈值和高阈值之间的为弱边缘
        weak = ((img >= low) & (img <= high))
        # 创建与原图同尺寸的结果图像，初始为全 0（即全黑）
        result = np.zeros_like(img, dtype=np.uint8)
        # 将强边缘位置赋值为 1（表示保留）
        result[strong] = 1
        # 遍历除边界以外的所有像素，检查弱边缘是否与强边缘相邻
        for i in range(1, img.shape[0] - 1):  # 图像的高（行数）
            for j in range(1, img.shape[1] - 1):  # 图像的宽（列数）
                # 如果当前像素是弱边缘，且其 3x3 邻域中包含强边缘
                if weak[i, j] and np.any(strong[i - 1:i + 2, j - 1:j + 2]):
                    # 则认为它也是有效边缘，设置为 1（保留）
                    result[i, j] = 1
        # 返回最终的边缘图：强边缘和与强边缘相连的弱边缘为1，其余为0
        return result


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageFilter(root)
    root.mainloop()
