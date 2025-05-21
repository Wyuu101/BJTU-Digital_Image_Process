import tkinter as tk
from tkinter import Menu, Listbox, Text, Scrollbar, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from itertools import combinations

photo = None
resize_photo = None
resize_photo_mask = None
photo_gray = None
resize_photo_gray = None
card_list = []
card_mask_list = []
card_class_list = []
highlighted_images = []
valid_set_list = []
global listbox



def open_image():
    global photo,resize_photo
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if filepath:
        #清除缓存
        clear_cache()
        img = Image.open(filepath)
        photo = np.array(img)


        img = img.resize((500, 400))  # 调整图片大小以适应画布
        resize_photo = np.array(img)
        display_image(resize_photo)
        btn.config(state=tk.NORMAL)

def display_image(image):
    canvas.delete("all")
    pil_img = Image.fromarray(image)  # numpy array 转 PIL Image
    tk_img = ImageTk.PhotoImage(pil_img)  # PIL Image 转 Tkinter PhotoImage
    canvas.image = tk_img  # 保持引用
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

def clear_cache():
    global photo,resize_photo,resize_photo_mask,photo_gray,resize_photo_gray,card_list,card_mask_list,card_class_list,highlighted_images,listbox,valid_set_list
    photo = None
    resize_photo = None
    resize_photo_mask = None
    photo_gray = None
    resize_photo_gray = None
    card_list = []
    card_mask_list = []
    card_class_list = []
    highlighted_images = []
    valid_set_list = []
    listbox.delete(0, tk.END)

def pre_process_image():
    global photo, resize_photo, resize_photo_gray
    resize_photo_gray = cv2.cvtColor(resize_photo, cv2.COLOR_RGB2GRAY)



    # 预处理：模糊 + 边缘检测
    blurred = cv2.GaussianBlur(resize_photo, (5, 5), 0)


    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                # 排序四个角点
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # 透视变换获取抠图
                width, height = 200, 300
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(resize_photo, M, (width, height))
                card_list.append(warped)
                # ===== 生成并保存该牌在原图中的掩码 =====
                mask = np.zeros(resize_photo_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                card_mask_list.append(mask)
                # mask_filename = f"card_mask_{card_count}.png"
                # cv2.imwrite(mask_filename, mask)
                card_count += 1

class Card:
    def __init__(self, amount, color, shape, fill,mask):
        self.amount = amount
        self.color = color
        self.shape = shape
        self.fill = fill
        self.mask = mask

def detect_shapes_and_properties(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #_, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)
    #_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 自适应阈值（局部处理）
    thresh = cv2.adaptiveThreshold(
        blur,  # 输入图像（灰度）
        255,  # 最大值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 或 MEAN_C
        cv2.THRESH_BINARY_INV,  # 类型（可改为 THRESH_BINARY）
        11,  # 邻域大小（奇数，影响局部范围）
        2  # 常量C，调节明暗偏移
    )
    # 轮廓提取
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    symbol_count = 0
    shape_types = []
    colors = []
    fillings = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue  # 忽略小轮廓（噪声）
        symbol_count += 1


        # 形状分析
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 7700:
            shape = "椭圆形"
        elif area < 5900:
            shape = "菱形"
        else:
            shape = "波浪形"
        shape_types.append(shape)
        foreground = cv2.bitwise_and(rgb, rgb, mask=thresh)
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        non_black_mask = np.any(foreground != [0, 0, 0], axis=-1)
        hue_channel = hsv[:, :, 0]
        h_values = hue_channel[non_black_mask]
        h_value = np.mean(h_values)

        if 0 <= h_value <= 35 or h_value >= 160:
            color = "红色"
        elif 35 < h_value < 90:
            color = "绿色"
        elif 90 <= h_value < 160:
            color = "蓝色"
        colors.append(color)
        # 填充分析
        symbol_mask = mask[y:y + h, x:x + w]
        symbol_gray = gray[y:y + h, x:x + w]
        mean_inside = cv2.mean(symbol_gray, mask=symbol_mask)[0]
        if mean_inside > 190:
            filling = "空心"
        elif mean_inside < 130:
            filling = "填满"
        else:
            filling = "竖线"
        fillings.append(filling)

    a= {
        "amount": symbol_count,
        "color": max(set(colors), key=colors.count),
        "shape": max(set(shape_types), key=shape_types.count),
        "fill": max(set(fillings), key=fillings.count)
    }
    return a

def feature_extraction():
    global card_list,card_mask_list,card_class_list
    for i in range(len(card_list)):
        feature = detect_shapes_and_properties(card_list[i])
        card_instance = Card(feature["amount"],feature["color"],feature["shape"],feature["fill"],card_mask_list[i])
        card_class_list.append(card_instance)


def is_valid_set(c1, c2, c3):
    # 对每个属性判断是否要么全相同，要么全不同
    attrs = ['amount', 'color', 'shape', 'fill']
    for attr in attrs:
        values = {getattr(c1, attr), getattr(c2, attr), getattr(c3, attr)}
        if len(values) == 2:  # 有两种值，不合法
            return False
    return True

def find_all_valid_sets(cards):
    valid_sets = []
    for c1, c2, c3 in combinations(cards, 3):
        if is_valid_set(c1, c2, c3):
            valid_sets.append((c1, c2, c3))
    return valid_sets


def draw_mask_outline(original_img, mask, color=(255,0, 0), thickness=3):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = original_img.copy()
    cv2.drawContours(outlined, contours, -1, color, thickness)
    return outlined



def mainProc():
    global resize_photo,valid_set_list
    btn.config(state=tk.DISABLED)
    pre_process_image()
    feature_extraction()
    valid_sets = find_all_valid_sets(card_class_list)
    for idx, valid_set in enumerate(valid_sets):
        c1, c2, c3 = valid_set
        # 合并三个 mask 做展示
        combined_mask = c1.mask | c2.mask | c3.mask
        highlighted_image = draw_mask_outline(resize_photo, combined_mask)
        highlighted_images.append(highlighted_image)
        valid_set_list.append(valid_set)
        listbox.insert(tk.END, f"第{idx+1}个合法Set")


def on_listbox_select(_):
    global valid_set_list
    selected_indices = listbox.curselection()  # 当前选中的项的索引（可能是多个）
    if selected_indices:
        index = selected_indices[0]
        value = listbox.get(index)
        display_image(highlighted_images[index])
        c1, c2, c3 = valid_set_list[index]
        text_box.insert(tk.END, "\n")
        text_box.insert(tk.END, f"{value}:\n")
        text_box.insert(tk.END, f"卡牌1 - 数量: {c1.amount}, 颜色: {c1.color}, 形状: {c1.shape}, 填充: {c1.fill}\n")
        text_box.insert(tk.END, f"卡牌2 - 数量: {c2.amount}, 颜色: {c2.color}, 形状: {c2.shape}, 填充: {c2.fill}\n")
        text_box.insert(tk.END, f"卡牌3 - 数量: {c3.amount}, 颜色: {c3.color}, 形状: {c3.shape}, 填充: {c3.fill}\n")
        text_box.see(tk.END)

# 主窗口
root = tk.Tk()
root.title("形色牌Set查找")
root.geometry("800x600")  # 设置窗口大小

# ------------------- 第一行：菜单栏 -------------------
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X)

# 使用 Menu 创建一个 menubar（仍然需要放入 root）
menubar = Menu(top_frame, tearoff=0)
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="打开图片", command=open_image)
file_menu.add_command(label="退出", command=root.quit)
menubar.add_cascade(label="文件", menu=file_menu)

# 创建一个变量用于记录当前选中的选项
selected_option = tk.StringVar(value="1")
selected_option.get()
# 创建菜单并添加单选项
mode_menu = Menu(menubar, tearoff=0)
mode_menu.add_radiobutton(label="第一批图片", variable=selected_option, value="1")
mode_menu.add_radiobutton(label="第二批图片", variable=selected_option, value="2")
mode_menu.add_radiobutton(label="第三批图片", variable=selected_option, value="3")

# 添加到菜单栏
menubar.add_cascade(label="批次选择", menu=mode_menu)

# 把菜单绑定到 root
root.config(menu=menubar)

# 创建一个按钮（与菜单同一行）
btn = tk.Button(top_frame, text="开始", command=mainProc)
btn.config(state=tk.DISABLED)
btn.pack(side=tk.LEFT, padx=10)


# ------------------- 第二行：画布和显示列表 -------------------
# 创建一个中部 Frame 用于画布和列表并列放置
mid_frame = tk.Frame(root)
mid_frame.pack(fill=tk.BOTH, expand=True)

# 画布（左边）
canvas = tk.Canvas(mid_frame, bg="white", width=500, height=400)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# 显示列表（右边）
list_frame = tk.Frame(mid_frame)
list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

listbox = Listbox(list_frame, width=30)
listbox.bind("<<ListboxSelect>>", on_listbox_select)
scrollbar = Scrollbar(list_frame, orient="vertical", command=listbox.yview)
listbox.config(yscrollcommand=scrollbar.set)

listbox.pack(side=tk.LEFT, fill=tk.Y)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# ------------------- 第三行：文本提示框 -------------------
text_frame = tk.Frame(root)
text_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

text_label = tk.Label(text_frame, text="提示：")
text_label.pack(side=tk.LEFT)

text_box = Text(text_frame, height=10)
text_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

# 示例提示信息
text_box.insert(tk.END, "打开图片，点击开始以查找。")

root.mainloop()