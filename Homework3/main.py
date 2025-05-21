import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei，支持中文
matplotlib.use("TkAgg")  # 使用 Tkinter 作为 Matplotlib 后端

# 创建 Tkinter 窗口
gui = tk.Tk()
gui.title("数字图像处理实验三 直方图绘制")


# 设置全局变量 用于存储图像和绘制直方图的画布
img_label = None
canvas_widget = None


# 提取灰度图的灰度直方图数据
def gray_histogram(image_path):
    image = Image.open(image_path)
    image_gray = image.convert("L")  # 转换为灰度图
    image_gray_array = np.array(image_gray)  # 将灰度图转为二维数组
    # 展开二维数组，提取0-255每个灰度出现的频率，返回一个长度为256的数组
    histogram_data_total = np.histogram(image_gray_array.flatten(), bins=256, range=[0, 256])[0]  
    # 计算每个灰度值的出现频率
    histogram_data_frequency = []
    for value in histogram_data_total:
        histogram_data_frequency.append(value / image_gray_array.size)
    return histogram_data_frequency, image


# 提取彩色图片的三通道灰度直方图数据
def color_histogram(image_path):
    image = Image.open(image_path)
    if image.mode == "L":  # 检测到是灰度图
        # 弹出错误对话框，提示用户图片类型不符合
        messagebox.showerror("错误", "图片类型不符合：请打开彩色图片。")
        return None  # 终止后续操作，不进行处理
    image_color = image.convert("RGB")  # 确保读取进来的图片是彩色图片
    image_color_array = np.array(image_color)  # 将彩色图片转化为三维数组
    # 提取红色通道灰度数据
    histogram_data_total_r, _ = np.histogram(image_color_array[:, :, 0].flatten(), bins=256, range=[0, 256])
    # 提取绿色通道灰度数据
    histogram_data_total_g, _ = np.histogram(image_color_array[:, :, 1].flatten(), bins=256, range=[0, 256])
    # 提取蓝色通道灰度数据
    histogram_data_total_b, _ = np.histogram(image_color_array[:, :, 2].flatten(), bins=256, range=[0, 256])
    histogram_data_frequency_r = []
    histogram_data_frequency_g = []
    histogram_data_frequency_b = []
    # 计算每个灰度值的出现频率
    for value in histogram_data_total_r:
        histogram_data_frequency_r.append(value / image_color_array[:, :, 0].size)
    for value in histogram_data_total_g:
        histogram_data_frequency_g.append(value / image_color_array[:, :, 0].size)
    for value in histogram_data_total_b:
        histogram_data_frequency_b.append(value / image_color_array[:, :, 0].size)
    return histogram_data_frequency_r, histogram_data_frequency_g, histogram_data_frequency_b, image


# 显示已被打开的图片
def show_image(image):
    global img_label
    img = ImageTk.PhotoImage(image.resize((200, 200)))  # 缩放图片
    if img_label is None:
        img_label = tk.Label(gui, image=img)
        img_label.image = img
        img_label.pack(side=tk.LEFT, padx=10)  # 显示在左侧
    else:
        img_label.configure(image=img)
        img_label.image = img


# 显示已被打开图像的直方图
def show_histogram(histogram_data_frequency, is_gray=True):
    global canvas_widget
    fig = Figure(figsize=(7, 8), dpi=100)  # 创建 Matplotlib Figure

    if is_gray:
        ax = fig.add_subplot(111)
        ax.bar(range(256), histogram_data_frequency, color='gray')  # 直方图
        ax.set_title("灰度图-单通道灰度直方统计图")
    else:
        histogram_data_frequency_r, histogram_data_frequency_g, histogram_data_frequency_b = histogram_data_frequency
        ax_r = fig.add_subplot(311)
        ax_r.bar(range(256), histogram_data_frequency_r, color='red', alpha=0.5, label='红色通道灰度频率')
        ax_r.set_title("彩色图-三通道灰度直方统计图")
        ax_r.set_ylabel("统计频率")  # 设置纵坐标标签
        ax_r.legend()  # 自动添加图例

        ax_g = fig.add_subplot(312)
        ax_g.bar(range(256), histogram_data_frequency_g, color='green', alpha=0.5, label='绿色通道灰度频率')
        ax_g.set_ylabel("统计频率")  # 设置纵坐标标签
        ax_g.legend()  # 自动添加图例

        ax_b = fig.add_subplot(313)
        ax_b.bar(range(256), histogram_data_frequency_b, color='blue', alpha=0.5, label='蓝色通道灰度频率')
        ax_b.set_xlabel("灰度值")  # 设置横坐标标签
        ax_b.set_ylabel("统计频率")  # 设置纵坐标标签
        ax_b.legend()  # 自动添加图例

    # 检查canvas_widget是否已经存在（即之前已经创建过画布）
    if canvas_widget:
        # 如果存在，销毁当前的canvas_widget
        canvas_widget.get_tk_widget().destroy()
    # 创建一个新的FigureCanvasTkAgg对象，将Matplotlib图形fig嵌入到Tkinter窗口gui中
    canvas_widget = FigureCanvasTkAgg(fig, master=gui)
    # 绘制图形，将图形渲染到画布上
    canvas_widget.draw()
    # 将新的画布小部件加入到 Tkinter 窗口中，位置设置为右侧并添加10像素的内边距
    canvas_widget.get_tk_widget().pack(side=tk.RIGHT, padx=10)


# 打开图片文件，显示图片并且进行灰度直方数据提取和显示
def open_file(is_gray=True):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    # 判断是否是有效文件路径
    if file_path:
        try:
            if is_gray:
                histogram_data_frequency, image = gray_histogram(file_path)
                show_image(image)
                show_histogram(histogram_data_frequency, is_gray=True)
            else:
                result = color_histogram(file_path)  # 调用 color_histogram 函数
                if result is None:  # 如果返回值是 None，则说明图片是灰度图
                    return  # 终止后续操作，不再继续执行
                histogram_data_frequency_r, histogram_data_frequency_g, histogram_data_frequency_b, image = color_histogram(file_path)
                show_image(image)
                show_histogram((histogram_data_frequency_r, histogram_data_frequency_g, histogram_data_frequency_b), is_gray=False)
        # 添加文件路径异常捕获
        except ValueError as e:
            messagebox.showerror("文件路径错误", str(e))


# 创建打开灰度图片的按钮
btn_gray = tk.Button(gui, text="打开灰度图片", command=lambda: open_file(True))
btn_gray.pack(pady=10)

# 创建打开彩色图片的按钮
btn_color = tk.Button(gui, text="打开彩色图片", command=lambda: open_file(False))
btn_color.pack(pady=10)

# 运行 Tkinter 主循环
gui.mainloop()
