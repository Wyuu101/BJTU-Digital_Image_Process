import tkinter as tk  # 导入 Tkinter，用于创建 GUI
from tkinter import filedialog, messagebox  # 文件选择和消息框
from tkinter import ttk  # 增强版的 Tkinter 控件
from PIL import Image, ImageTk  # 用于图像加载和显示
import numpy as np  # 数组处理
import cv2  # OpenCV 图像处理库
from sklearn.cluster import KMeans  # 导入 sklearn 中的 KMeans 聚类算法


# 自定义 K-means 聚类函数
def custom_kmeans(data, k, max_iter=100):
    np.random.seed(42)
    # 从数据中随机选 k 个初始质心
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iter):
        # 计算每个点到每个质心的距离
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        # 为每个点分配最近的质心的标签
        labels = np.argmin(distances, axis=1)
        # 计算新的质心（每类的平均值）
        new_centroids = np.array([
            data[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        # 如果质心不再变化，提前结束迭代
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids  # 返回标签和质心


# GUI 主类
class KMeansSegmentationGUI:
    def __init__(self, root):
        self.root = root  # 主窗口
        self.root.title("数字图像处理实验十 基于机器学习的图像分割")  # 设置窗口标题
        self.image = None  # 当前加载的图像对象
        self.image_array_original = None  # 图像的数组数据
        self.image_array_proc = None # 自定义分割Kmeans函数的处理结果
        self.image_array_sk = None # sklearn库的Kmeans处理结果
        self.image_original = None
        self.image_proc = None
        self.image_sk = None
        self.k = 0 # 缓存K值
        self.setup_ui()  # 创建界面控件

    # 构建界面控件
    def setup_ui(self):
        frame = ttk.Frame(self.root)  # 创建一个控件容器
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=0)
        # 加载图像按钮
        ttk.Button(frame, text="加载图像", command=self.load_image).grid(row=0, column=0, padx=5)

        # K 值输入框和标签
        self.kmeans_label = ttk.Label(frame, text="聚类数K:")
        self.kmeans_label.grid(row=0, column=1)
        self.kmeans_label.config(state=tk.DISABLED)

        self.k_entry = ttk.Entry(frame, width=5)
        self.k_entry.insert(0, "4")  # 默认聚类数为4
        self.k_entry.grid(row=0, column=2, padx=5)
        self.k_entry.config(state=tk.DISABLED)

        # K-means分割按钮
        self.kmeans_button = ttk.Button(frame, text="K-Means分割", command=self.segment_custom)
        self.kmeans_button.grid(row=0, column=3, padx=5)
        self.kmeans_button.config(state=tk.DISABLED)

        # 创建一个变量，用于表示是否启用与sklearn KMeans对比
        self.compare_with_sklearn = tk.IntVar()

        # 添加与sklearn KMeans对比的勾选框
        self.check_box = ttk.Checkbutton(frame, text="与sklearn KMeans分割进行对比", variable=self.compare_with_sklearn,
                                                command=self.on_check)
        self.check_box.grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        self.check_box.config(state=tk.DISABLED)


        # 画布（用于显示原始图像）
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        self.label_canvas = tk.Label(self.root, text="原始图像")
        self.label_canvas.grid(row=2, column=0, padx=10, pady=10)

        # 画布（用于显示处理后的图像）
        self.canvas_new = tk.Canvas(self.root, bg="gray")
        self.canvas_new.grid(row=1, column=1, padx=10, pady=10)
        self.label_canvas_new = tk.Label(self.root, text="自定义KMeans分割结果")
        self.label_canvas_new.grid(row=2, column=1, padx=10, pady=10)

        # 画布（用于显示库函数处理后的图像）
        self.canvas_sk = tk.Canvas(self.root, bg="gray")
        self.canvas_sk.grid(row=1, column=2, padx=10, pady=10)
        self.canvas_sk.grid_forget()
        self.label_canvas_sk = tk.Label(self.root, text="sklearn KMeans分割结果")
        self.label_canvas_sk.grid(row=2, column=2, padx=10, pady=10)
        self.label_canvas_sk.grid_forget()

    # 当启用/禁用库函数对比
    def on_check(self):
        if self.compare_with_sklearn.get():
            self.canvas_sk.grid(row=1, column=2, padx=10, pady=10)
            self.label_canvas_sk.grid(row=2, column=2, padx=10, pady=10)
            if self.image_array_original is not None:
                # 避免重复调用库函数浪费资源
                if self.image_array_sk is not None:
                    if int(self.k_entry.get()) == self.k:
                        self.display_image(2)
                        return
            self.segment_sklearn()
        else:
            self.canvas_sk.grid_forget()
            self.label_canvas_sk.grid_forget()

    # 加载图像函数
    def load_image(self):
        file_path = filedialog.askopenfilename()  # 打开文件选择对话框
        if file_path:
            img = cv2.imread(file_path)  # 使用 OpenCV 读取图像
            if img is None:
                messagebox.showerror("错误", "无法打开图像")
                return
            self.image_array_original = img  # 保存图像数组
            self.image_array_original = cv2.cvtColor(self.image_array_original, cv2.COLOR_BGR2RGB)  # BGR 转 RGB 显示
            # 清除上一张原始图像
            self.canvas.delete("all")
            # 清除上一次操作结果
            self.canvas_new.delete("all")
            # 清除库函数处理结果
            self.canvas_sk.delete("all")
            self.display_image(0)
            self.image_array_proc = None
            self.display_image(1)
            self.image_array_sk = None
            self.display_image(2)
            self.kmeans_label.config(state=tk.NORMAL)
            self.k_entry.config(state=tk.NORMAL)
            self.kmeans_button.config(state=tk.NORMAL)
            self.check_box.config(state=tk.NORMAL)

    def display_image(self, place):
        # 显示原始图像
        if place == 0:
            if self.image_array_original is not None:
                pil_image = Image.fromarray(self.image_array_original).resize((400, 400))
                width, height = pil_image.size
                self.image_original = ImageTk.PhotoImage(pil_image)
                # 在画布上显示图片
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_original)
                # 调整画布自适应图片大小
                self.canvas.config(height=height, width=width)
        # 显示分割后的图像
        elif place == 1:
            if self.image_array_proc is not None:
                pil_image = Image.fromarray(self.image_array_proc).resize((400, 400))
                width, height = pil_image.size
                self.image_proc = ImageTk.PhotoImage(pil_image)
                self.canvas_new.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
                self.canvas_new.config(height=height, width=width)
        elif place == 2:
            if self.image_array_sk is not None:
                pil_image = Image.fromarray(self.image_array_sk).resize((400, 400))
                width, height = pil_image.size
                self.image_sk = ImageTk.PhotoImage(pil_image)
                self.canvas_sk.create_image(0, 0, anchor=tk.NW, image=self.image_sk)
                self.canvas_sk.config(height=height, width=width)
                # 缓存
                self.image_sk_last_time = self.image_array_sk

    # 使用自定义K-means进行图像分割
    def segment_custom(self):
        if self.image_array_original is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        try:
            k = int(self.k_entry.get())  # 获取K值

        except ValueError:
            messagebox.showerror("错误", "K值必须是整数")
            return

        img = self.image_array_original

        is_gray = len(img.shape) == 2 or img.shape[2] == 1  # 判断是否灰度图
        # 展平图像为二维特征数组
        if is_gray:
            flat_img = img.reshape((-1, 1)).astype(np.float32)
        else:
            flat_img = img.reshape((-1, 3)).astype(np.float32)


        # 进行聚类
        labels, centroids = custom_kmeans(flat_img, k)
        # 将标签重建为图像
        self.image_array_proc = centroids[labels].reshape(img.shape)
        if not is_gray:
            self.image_array_proc = cv2.cvtColor(self.image_array_proc.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转为RGB显示
        self.display_image(1)

        if self.compare_with_sklearn.get():
            # 在后台提前绘制好sklearn库函数的处理结果
            self.segment_sklearn()

    # 使用 sklearn 的 KMeans 分割图像
    def segment_sklearn(self):
        if self.image_array_original is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        try:
            k = int(self.k_entry.get())  # 获取K值
            self.k = k
        except ValueError:
            messagebox.showerror("错误", "K值必须是整数")
            return

        img = self.image_array_original
        is_gray = len(img.shape) == 2 or img.shape[2] == 1  # 是否灰度图
        # 转换为一维特征向量
        if is_gray:
            flat_img = img.reshape((-1, 1)).astype(np.float32)
        else:
            flat_img = img.reshape((-1, 3)).astype(np.float32)

        model = KMeans(n_clusters=k, n_init=10)  # 创建 KMeans 模型
        labels = model.fit_predict(flat_img)  # 聚类
        self.image_array_sk = model.cluster_centers_[labels].reshape(img.shape)  # 根据标签重建图像
        if not is_gray:
            self.image_array_sk = cv2.cvtColor(self.image_array_sk.astype(np.uint8), cv2.COLOR_BGR2RGB)
        self.display_image(2)


# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()  # 创建主窗口
    app = KMeansSegmentationGUI(root)  # 实例化应用
    root.mainloop()  # 运行 GUI 主循环
