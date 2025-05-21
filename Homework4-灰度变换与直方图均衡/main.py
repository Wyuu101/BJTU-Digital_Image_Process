import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, Frame, Label, Button, Scale,messagebox
from PIL import Image, ImageTk
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei，支持中文
import matplotlib
matplotlib.use("TkAgg")  # 使用 Tkinter 作为 Matplotlib 后端

# 全局变量存储文件路径
file_path = None
original_image_array= None
canvas_widget = None
canvas_original_widget = None


# 反色变换
def negative_transform(image):
    if image is not None:
        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)
        # 通过循环对每一个图像的像素点一一进行操作
        for i in range(height):
            for j in range(width):
                new_image[i, j] = 255 - image[i, j]
        display_image(canvas,new_image)
        show_histogram(gray_histogram(new_image),2,1,"变换后图像灰度直方图",False)
    else:
        messagebox.showerror("错误","请先打开一张图片")

# 对数变换(image为原图片，c为乘积因子)
def log_transform(image, c=1.0):
    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)
    # 通过循环对每一个图像的像素点一一进行操作
    for i in range(height):
        for j in range(width):
            transformed_value = c * math.log(int(image[i, j])+1)
            # 使用min确保灰度值在255以内
            new_image[i, j] = min(int(transformed_value), 255)
    return new_image

# 幂次变换
def gamma_transform(image, c=1.0, gamma=1.0):
    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)
    # 通过循环对每一个图像的像素点一一进行操作
    for i in range(height):
        for j in range(width):
            # 使用min确保灰度值在255以内
            transformed_value = c * pow(image[i, j], gamma)
            new_image[i, j] = min(int(transformed_value), 255)
    return new_image

# 直方图均衡化
def histogram_equalization(image):
    if image is not None:
        height, width = image.shape
        # 计算常数 GL/SZ
        c = 255 / (height * width)
        # 将二维数组变为一维数组便于操作
        flat_array = np.ravel(image)
        # 用于统计原图像灰度直方图数据
        gray_histogram_list = np.zeros(256, dtype=int)
        # 遍历像素点并统计灰度直方图
        for index, element in enumerate(flat_array):
            gray_histogram_list[element] += 1
        # 计算分布累计函数
        sum = np.cumsum(gray_histogram_list)
        # 创建一个灰度映射列表
        map = np.zeros(256, dtype=np.uint8)
        # 遍历分布累计函数值
        for index, element in enumerate(sum):
            # 图像均衡算法
            map[index] = min(round(c * element), 255)
        for index, element in enumerate(flat_array):
            # 将旧图像的灰度进行重新映射，同时避免超过255
            flat_array[index] = min(map[flat_array[index]], 255)
        new_image = flat_array.reshape(height, width)
        #处理算法已结束，以下是相关显示图像的操作
        display_image(canvas,new_image)
        show_histogram(gray_histogram(new_image),2,1,"变换后图像灰度直方图",False)
    else:
        messagebox.showerror("错误","请先打开一张图片")





# 打开图像
def open_image():
    #全局变量，用于存储图片路径和图片数组格式
    global file_path, original_image_array,canvas_widget
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('L')  # 确保是灰度图像
        original_image_array = np.array(img, dtype=np.uint8)  # 确保数组为uint8类型
        display_image(canvas_original, original_image_array)
        show_histogram(gray_histogram(original_image_array),2,0,"原始图像灰度直方图")



# 提取灰度图的灰度直方图数据
def gray_histogram(image_gray_array):
    # 展开二维数组，提取0-255每个灰度出现的频率，返回一个长度为256的数组
    histogram_data_total = np.histogram(image_gray_array.flatten(), bins=256, range=[0, 256])[0]
    # 计算每个灰度值的出现频率
    histogram_data_frequency = []
    for value in histogram_data_total:
        histogram_data_frequency.append(value / image_gray_array.size)
    return histogram_data_frequency

# 显示图像
def display_image(canvas, image_array):
    img = Image.fromarray(image_array).resize((300,300))
    # 将数组转化为可显示的图像对象
    img_display = ImageTk.PhotoImage(img)
    # 让画布自适应图像大小
    canvas.config(width=300, height=300)
    # 让图像显示在画布上，图像以画布的(0,0)坐标为左上角开始定位
    canvas.create_image(0, 0, anchor=tk.NW, image=img_display)
    canvas.image = img_display


# 显示原始图像的直方图
def show_histogram(histogram_data_frequency,row,column,title,is_original=True):
    global canvas_original_widget,canvas_widget
    fig = Figure(figsize=(4, 3), dpi=100)  # 创建 Matplotlib Figure
    ax = fig.add_subplot(111)
    ax.bar(range(256), histogram_data_frequency, color='gray')  # 直方图
    ax.set_title(title)
    widget = canvas_original_widget
    if is_original:
        pass
    else:
        widget = canvas_widget
    # 检查canvas_widget是否已经存在（即之前已经创建过画布）
    if widget:
        # 如果存在，销毁当前的widget
        widget.get_tk_widget().destroy()
    # 创建一个新的FigureCanvasTkAgg对象，将Matplotlib图形fig嵌入到Tkinter窗口gui中
    widget = FigureCanvasTkAgg(fig, master=gui)
    # 绘制图形，将图形渲染到画布上
    widget.draw()
    # 将新的画布小部件加入到 Tkinter 窗口中，位置设置为右侧并添加10像素的内边距
    widget.get_tk_widget().grid(row=row, column=column, padx=20, pady=10)


def update_gamma(val):
    #判定图像是否存在，避免引发报错
    if file_path:
        transformed_image = gamma_transform(original_image_array, c=1.0, gamma=float(val))
        display_image(canvas, transformed_image)
        show_histogram(gray_histogram(transformed_image),2,1,"变换后图像灰度直方图",False)

def update_log(val):
    # 判定图像是否存在，避免引发报错
    if file_path:
        transformed_image = log_transform(original_image_array, c=float(val))
        display_image(canvas,transformed_image)
        show_histogram(gray_histogram(transformed_image), 2, 1, "变换后图像灰度直方图", False)

def update_c(val):
    # 判定图像是否存在，避免引发报错
    if file_path:
        transformed_image = gamma_transform(original_image_array, c=float(val))
        display_image(canvas,transformed_image)
        show_histogram(gray_histogram(transformed_image), 2, 1, "变换后图像灰度直方图", False)

# GUI 界面
gui = tk.Tk()
gui.title("数字图像处理实验四 灰度变换")

label_original_image =Label(gui,text="原始图像", font=("楷体", 16))
label_original_image.grid(row=0, column=0, padx=20, pady=10)


label_image =Label(gui,text="变换后的图像", font=("楷体", 16))
label_image.grid(row=0, column=1, padx=20, pady=10)

label_operation = Label(gui,text="操作区", font=("楷体", 16))
label_operation.grid(row=0, column=2, padx=20, pady=10)

#创建用于显示原始图像的画布
canvas_original = Canvas(gui)
canvas_original.grid(row=1, column=0, padx=20, pady=10)

#创建用于显示变化后图像的画布
canvas = Canvas(gui)
canvas.grid(row=1, column=1, padx=20, pady=10)

#操作区（按钮和滑动条）
btn_frame = Frame(gui)
btn_frame.grid(row=1, column=2, padx=20, pady=10)

btn_open = Button(btn_frame, text="打开图像", width=8,height=1,font=("楷体", 16),command=open_image)
btn_open.grid(row=0, column=0, padx=20, pady=10)

btn_negative = Button(btn_frame, text="反色变换", width=8,height=1,font=("楷体", 16), command=lambda: negative_transform(original_image_array))
btn_negative.grid(row=1, column=0, padx=20, pady=10)

btn_hist_eq = Button(btn_frame, text="直方图均衡化", width=8,height=1,font=("楷体", 16), command=lambda: histogram_equalization(original_image_array))
btn_hist_eq.grid(row=2, column=0, padx=20, pady=10)

#拖动条区域
scale_frame = Frame(gui)
scale_frame.grid(row=2, column=2, padx=20, pady=10)

log_scale = Scale(scale_frame, from_=1, to=50, length=400, orient=tk.HORIZONTAL, label="对数变换参数-c", command=update_log)
log_scale.set(10) # 默认值为10
log_scale.grid(row=0, column=0, padx=20, pady=10)

gamma_scale = Scale(scale_frame, from_=0, to=2, length=400, orient=tk.HORIZONTAL, resolution=0.1, label="幂数变换参数-γ", command=update_gamma)
gamma_scale.set(1)  # 默认gamma值为1
gamma_scale.grid(row=1, column=0, padx=20, pady=10)

c_scale = Scale(scale_frame, from_=0, to=5, length=400,  resolution=0.2,orient=tk.HORIZONTAL, label="幂数变换参数-c", command=update_c)
c_scale.set(1)  # 默认c值为1
c_scale.grid(row=2, column=0, padx=20, pady=10)



gui.mainloop()
