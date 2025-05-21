import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import numpy as np
import matplotlib
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei，支持中文
matplotlib.use("TkAgg")  # 使用 Tkinter 作为 Matplotlib 后端


class ImageTransformer:
    def __init__(self, gui):
        self.label_var = None
        self.label_threshold = None
        self.gui = gui
        self.gui.title("数字图像处理实验六 基于阈值的图像分割")

        # 顶部按钮框架
        self.frame = ttk.Frame(gui)
        self.frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=0)

        # 信息提示区
        self.frame_label = ttk.Frame(self.gui)
        self.frame_label.grid_forget()

        # 按钮控件
        self.btn_load = ttk.Button(self.frame, text="加载图像", command=self.open_image)
        self.btn_load.grid(row=0, column=0, padx=5, pady=5)

        self.btn_otsu = ttk.Button(self.frame, text="大津分割", command=self.threshold_segmentation)
        self.btn_otsu.grid(row=0, column=1, padx=5, pady=5)
        self.btn_otsu.config(state=tk.DISABLED)

        self.label_hsv = ttk.Label(self.frame, text="HSV色彩分割")
        self.label_hsv.grid(row=0, column=2, padx=5, pady=5)
        self.label_hsv.config(state=tk.DISABLED)

        # 下拉菜单
        self.color_var = tk.StringVar(value="红色")
        self.color_menu = ttk.Combobox(self.frame, values=["红色", "蓝色", "黄色", "自定义"])
        self.color_menu.grid(row=0, column=4, padx=5, pady=5)
        self.color_menu.config(state=tk.DISABLED)
        # 将选择事件绑定到对应的函数
        self.color_menu.bind("<<ComboboxSelected>>", self.on_color_menu_changed)

        self.upper_limit_scale = Scale(self.frame, from_=0, to=359, length=200, orient=tk.HORIZONTAL, label="阈值上限", command=self.update_scale)
        self.upper_limit_scale.grid(row=0, column=6, padx=5, pady=5)
        self.upper_limit_scale.grid_forget()

        self.lower_limit_scale = Scale(self.frame, from_=0, to=359, length=200, orient=tk.HORIZONTAL, label="阈值下限", command=self.update_scale)
        self.lower_limit_scale.grid(row=0, column=5, padx=5, pady=5)
        self.lower_limit_scale.grid_forget()

        # 画布（用于显示原始图像）
        self.canvas = tk.Canvas(gui, bg="gray")
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        # 画布（用于显示分割后的图像）
        self.canvas_new = tk.Canvas(gui, bg="gray")
        self.canvas_new.grid(row=1, column=1, padx=10, pady=10)

        # 画布（用于显示灰度直方图）
        self.fig = Figure(figsize=(4, 3), dpi=100)  # 创建 Matplotlib Figure
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.gui)

        # 配置 grid 行列权重，使画布可以自动扩展
        self.gui.grid_rowconfigure(1, weight=1)
        self.gui.grid_columnconfigure(0, weight=1)

    image_array_original = None
    image_tk_original = None
    image_array_new = None
    image_tk_new = None
    # hsv色相分割阈值上下限
    upper_limit = 30
    lower_limit = 0

    # 当下拉选框的选择更改时调用的函数
    def on_color_menu_changed(self, _):
        selected_value = self.color_menu.get()  # 获取当前选择的值
        if selected_value != "自定义":
            if selected_value == "红色":
                self.upper_limit = 15
                self.lower_limit = 345
            elif selected_value == "黄色":
                self.lower_limit = 45
                self.upper_limit = 60
            elif selected_value == "蓝色":
                self.lower_limit = 195
                self.upper_limit = 225
            self.threshold_color_segmentation()
            # 隐藏滑动条
            self.upper_limit_scale.grid_forget()
            self.lower_limit_scale.grid_forget()
        else:
            # 显示滑动条
            self.upper_limit_scale.grid(row=0, column=6, padx=5, pady=5)
            self.lower_limit_scale.grid(row=0, column=5, padx=5, pady=5)

    def update_scale(self, _):
        self.lower_limit = int(self.lower_limit_scale.get())
        self.upper_limit = int(self.upper_limit_scale.get())
        self.threshold_color_segmentation()

    # 打开图像（支持通用格式以及raw文件
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.jpeg")])
        if not file_path:
            return
        try:
            self.image_array_original = np.array(Image.open(file_path), dtype=np.uint8)
            # 判断图片类型-RGB图片
            if self.image_array_original.ndim == 3 and self.image_array_original.shape[2] == 3:
                # 禁用大津分割
                self.btn_otsu.config(state=tk.DISABLED)
                # 启用色彩分割
                self.label_hsv.config(state=tk.NORMAL)
                # 启用下拉选框
                self.color_menu.config(state=tk.NORMAL)
                # 隐藏灰度直方图区域
                self.canvas_widget.get_tk_widget().grid_forget()
                # 隐藏信息提示区
                self.frame_label.grid_forget()
                # 清除上一次分割结果
                self.canvas_new.delete("all")
            else:
                # 启用大津分割
                self.btn_otsu.config(state=tk.NORMAL)
                # 禁用色彩分割
                self.label_hsv.config(state=tk.DISABLED)
                # 禁用下拉选框
                self.color_menu.config(state=tk.DISABLED)
                # 隐藏滑动条
                self.upper_limit_scale.grid_forget()
                self.lower_limit_scale.grid_forget()
                # 清除上一次灰度直方图
                self.canvas_widget.figure.clear()
                # 显示灰度直方图区域
                self.canvas_widget.get_tk_widget().grid(row=2, column=0, padx=10, pady=10)
                # 清除上一次分割结果
                self.canvas_new.delete("all")

            self.display_image()
            self.image_array_new = None
            self.display_image(False)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图像: {str(e)}")

    def display_image(self, is_original=True):
        # 显示原始图像
        if is_original:
            if self.image_array_original is not None:
                pil_image = Image.fromarray(self.image_array_original)
                width, height = pil_image.size
                self.image_tk_original = ImageTk.PhotoImage(pil_image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk_original)
                self.canvas.config(height=height, width=width)
        # 显示分割后的图像
        else:
            if self.image_array_new is not None:
                pil_image = Image.fromarray(self.image_array_new)
                width, height = pil_image.size
                self.image_tk_new = ImageTk.PhotoImage(pil_image)
                self.canvas_new.create_image(0, 0, anchor=tk.NW, image=self.image_tk_new)
                self.canvas_new.config(height=height, width=width)

    # 显示阈值分割后的直方图
    def show_histogram(self, histogram_data_frequency, row, column, threshold, title):
        ax = self.fig.add_subplot(111)
        ax.bar(range(256), histogram_data_frequency, color='gray')  # 直方图
        ax.set_title(title)
        # 绘制阈值标记（红线）
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'threshold: {threshold}')
        # 绘制图形，将图形渲染到画布上
        self.canvas_widget.draw()
        # 将新的画布小部件加入到 Tkinter 窗口中，位置设置为右侧并添加10像素的内边距
        self.canvas_widget.get_tk_widget().grid(row=row, column=column, padx=20, pady=10)

    def threshold_segmentation(self):
        # 总像素数
        total_pixels = self.image_array_original.size
        # 灰度直方统计表
        gray_histogram_list = np.zeros(256)
        # 图像转一维数组
        flat_array = self.image_array_original.ravel()
        # 遍历像素点并统计灰度直方图
        for index, element in enumerate(flat_array):
            gray_histogram_list[element] += 1
        # 声明最大方差变量
        max_var = 0
        threshold = 1
        # 开始寻找方差最大时的阈值
        for i in range(1, 255):
            # C0组的概率
            w0 = np.sum(gray_histogram_list[0:i+1]) / total_pixels
            # C1组的概率
            w1 = 1-w0
            sum0 = 0
            sum1 = 0
            for index, element in enumerate(gray_histogram_list):
                if index <= i:
                    sum0 += index * element
                else:
                    sum1 += index * element
            if total_pixels * w0 == 0 or total_pixels * w1 == 0:
                continue
            # C0组灰度均值
            u0 = sum0 / (total_pixels * w0)
            # C1组灰度均值
            u1 = sum1 / (total_pixels * w1)
            # 图像平均灰度级
            u = u0*w0+u1*w1
            # 取均方值最大
            max_var_old = max_var
            max_var = max(max_var, w0*pow((u-u0), 2)+w1*pow((u-u1), 2))
            # 记录下对应的阈值
            if max_var != max_var_old:
                threshold = i
        # 绘制分割后的图片(默认全白）
        image_flat_array_new = np.full(total_pixels, 255, dtype=np.uint8)
        for i in range(total_pixels):
            if flat_array[i] > threshold:
                image_flat_array_new[i] = 0
        self.image_array_new = image_flat_array_new.reshape(self.image_array_original.shape)
        self.display_image(False)
        self.show_histogram(gray_histogram_list,2,0,threshold, "灰度直方图")
        # 信息提示框架
        self.frame_label.grid(row=2, column=1, padx=10, pady=10)
        self.label_threshold = ttk.Label(self.frame_label, text=f'最大方差阈值:{threshold}', font=("楷体", 16))
        self.label_threshold.grid(row=0, column=0, padx=10, pady=10)
        self.label_var = ttk.Label(self.frame_label, text=f'方差:{round(max_var, 2)}', font=("楷体", 16))
        self.label_var.grid(row=1, column=0, padx=10, pady=10)

    # 彩色图像阈值分割
    def threshold_color_segmentation(self):
        hsv_image = cv2.cvtColor(self.image_array_original, cv2.COLOR_RGB2HSV)
        # 提取色相通道（H）
        hue_channel = hsv_image[:, :, 0]
        # 映射到360色盘
        hue_channel_360 = hue_channel * 2
        # 创建掩码
        if self.lower_limit <= self.upper_limit:
            mask = (hue_channel_360 >= self.lower_limit) & (hue_channel_360 <= self.upper_limit)
        else:
            mask = (hue_channel_360 >= self.lower_limit) | (hue_channel_360 <= self.upper_limit)
        # 创建分割后的图像模板
        self.image_array_new = np.zeros_like(self.image_array_original)
        # 显示分割后的图片
        self.image_array_new[mask] = self.image_array_original[mask]
        self.display_image(False)


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageTransformer(root)
    root.mainloop()
