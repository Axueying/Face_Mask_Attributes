import tkinter as tk
import cv2
import time
from PIL import Image, ImageTk
import json
import oss2
from viapi.fileutils import FileUtils
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkfacebody.request.v20191230.DetectMaskRequest import DetectMaskRequest
from aliyunsdkfacebody.request.v20191230.RecognizeFaceRequest import RecognizeFaceRequest
from aliyunsdkfacebody.request.v20191230.RecognizeExpressionRequest import RecognizeExpressionRequest

# 阿里云的AccessKey和Secert(处于隐私和安全考虑，已省去)
AccessKey = 'LTAI4G5oCUcAbm2UyhtaRQZ8'
Secert = 'ccbyw5i1LNSo2SFwftqunZyn2Xw8gv'
# 阿里云OSS对象存储的Bucket名字
BucketName = 'py-face-images'

# OpenCV分类器的完整路径
face_classfier_path = "haarcascade_frontalface_alt.xml"
eye_classifer_path = "haarcascade_eye_tree_eyeglasses.xml"

# 实例化阿里云的各种服务
auth = oss2.Auth(AccessKey, Secert)
bucket = oss2.Bucket(auth, 'http://oss-cn-shanghai.aliyuncs.com', BucketName)
file_utils = FileUtils(AccessKey, Secert)
client = AcsClient(AccessKey, Secert, 'cn-shanghai')

# 开启摄像头
camera = cv2.VideoCapture(0)

# 定义openCV的分类器
fontface_classfier = cv2.CascadeClassifier(face_classfier_path)  # 正脸识别
eye_classfier = cv2.CascadeClassifier(eye_classifer_path)  # 眼睛识别


# left_eye_classfier = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")#左眼识别

# 显示摄像头实时画面函数
######################################################################################
def video_loop():
    # 识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    color = (0, 255, 0)
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        # 将当前桢图像转换成灰度图像
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = fontface_classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        eyesRects = eye_classfier.detectMultiScale(grey)
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                # 画出矩形框（绿色）
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

        if len(eyesRects) > 0:
            for (x, y, w, h) in eyesRects:
                # 画出眼睛的矩形方框（蓝色）
                frame = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        camera_panel.imgtk = imgtk
        camera_panel.config(image=imgtk)
        window.after(1, video_loop)


# 更新右侧信息栏的函数，点击刷新按钮执行的函数
###################################################################################
def info_update():
    # 用于存储图片的名字，图片对象，以及阿里云返回的url
    img_list = []
    # 时间对象，用于给每张图片赋予不同的名字
    time1 = time.localtime(time.time())
    # 从摄像头读取照片
    success, img = camera.read()
    if success:
        # 将当前桢图像转换成灰度图像
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = fontface_classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        # 眼睛检测
        eyesRects = eye_classfier.detectMultiScale(grey)

        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 通过人脸检测的矩形获取人脸图片
                image_item = img[y - 50: y + h + 50, x - 50: x + w + 50]
                image_item = cv2.resize(image_item, (100, 100), interpolation=cv2.INTER_LINEAR)
                img_name = "images\\" + str(time1.tm_year) + str(time1.tm_mon) + str(time1.tm_mday) + str(
                    time1.tm_hour) + str(time1.tm_min) + str(time1.tm_sec) + ".png"
                cv2.imwrite(
                    "images\\" + str(time1.tm_year) + str(time1.tm_mon) + str(time1.tm_mday) + str(time1.tm_hour) + str(
                        time1.tm_min) + str(time1.tm_sec) + ".png", image_item)
                # 上传到阿里云OSS服务的Bucket（该服务收费）同时获取图片的url
                bucket.put_object_from_file(img_name, img_name)
                oss_url = file_utils.get_oss_url(img_name, "png", True)
                # print(oss_url)

                img_info = []
                img_info.append(img_name)
                img_info.append(image_item)
                img_info.append(oss_url)

                img_list.append(img_info)
        # 如果没获取到人脸但是获取到了眼睛，通过眼睛位置捕获人脸（可能重复捕获）
        elif len(eyesRects) > 0:
            for eyesRect in eyesRects:
                x, y, w, h = eyesRect

                image_item = img[y - 130: y + h + 130, x - 130: x + w + 130]
                image_item = cv2.resize(image_item, (100, 100), interpolation=cv2.INTER_AREA)
                img_name = "images\\" + str(time1.tm_year) + str(time1.tm_mon) + str(time1.tm_mday) + str(
                    time1.tm_hour) + str(time1.tm_min) + str(time1.tm_sec) + ".png"

                cv2.imwrite(
                    "images\\" + str(time1.tm_year) + str(time1.tm_mon) + str(time1.tm_mday) + str(time1.tm_hour) + str(
                        time1.tm_min) + str(time1.tm_sec) + ".png", image_item)
                bucket.put_object_from_file(img_name, img_name)
                oss_url = file_utils.get_oss_url(img_name, "png", True)
                # print(oss_url)

                img_info = []
                img_info.append(img_name)
                img_info.append(image_item)
                img_info.append(oss_url)

                img_list.append(img_info)

        # 销毁上一次的当前视频中的人脸信息
        for item in frame_c_now.winfo_children():
            item.destroy()
        frame_c_now.configure(height=0)

        # 更新右边的当前信息和历史记录
        for item in img_list:
            # 0.是否佩戴口罩，1.颜色
            Mask_info = DetectMask(item[2])
            # 0.性别，1.年龄，2.颜值，3.眼镜，4.帽子
            attributes = face_attributes(item[2])
            # neutral（中性）、happiness（高兴）、surprise（惊讶）、sadness（伤心）、anger（生气）、disgust（厌恶）、fear（害怕）。
            expression = detect_experssions(item[2])

            # 更新右侧当前视频中人脸信息
            item_Frame = tk.Frame(frame_c_now)
            item_Frame_right = tk.Frame(item_Frame)
            img_lable = tk.Label(item_Frame)
            mask_lable = tk.Label(item_Frame_right, text="是否佩戴口罩：" + Mask_info[0], fg=Mask_info[1])
            sex_lable = tk.Label(item_Frame_right,
                                 text="性别：" + attributes[0] + "  年龄：" + str(attributes[1]) + "  颜值：" + str(
                                     attributes[2]))
            experssion_lable = tk.Label(item_Frame_right, text="表情：" + expression)
            glass_lable = tk.Label(item_Frame_right, text="是否佩戴眼镜：" + attributes[3])
            hat_lable = tk.Label(item_Frame_right, text="是否佩戴帽子：" + attributes[4])

            # 更新右侧历史记录人脸信息
            item_Frame_history = tk.Frame(frame_c_history)
            item_Frame_right_history = tk.Frame(item_Frame_history)
            img_lable_history = tk.Label(item_Frame_history)
            mask_lable_history = tk.Label(item_Frame_right_history, text="是否佩戴口罩：" + Mask_info[0], fg=Mask_info[1])
            sex_lable_history = tk.Label(item_Frame_right_history,
                                         text="性别：" + attributes[0] + "  年龄：" + str(attributes[1]) + "  颜值：" + str(
                                             attributes[2]))
            experssion_lable_history = tk.Label(item_Frame_right_history, text="表情：" + expression)
            glass_lable_history = tk.Label(item_Frame_right_history, text="是否佩戴眼镜：" + attributes[3])
            hat_lable_history = tk.Label(item_Frame_right_history, text="是否佩戴帽子：" + attributes[4])

            item[1] = cv2.cvtColor(item[1], cv2.COLOR_RGB2BGRA)
            current_image_item = Image.fromarray(item[1])
            imgtk = ImageTk.PhotoImage(image=current_image_item)
            img_lable.imgtk = imgtk
            img_lable.config(image=imgtk)
            item_Frame.pack()
            img_lable.pack(side="left")
            item_Frame_right.pack(side="right")
            mask_lable.pack()
            sex_lable.pack()
            experssion_lable.pack()
            glass_lable.pack()
            hat_lable.pack()

            img_lable_history.imgtk = imgtk
            img_lable_history.config(image=imgtk)
            item_Frame_history.pack()
            img_lable_history.pack(side="left")
            item_Frame_right_history.pack(side="right")
            mask_lable_history.pack()
            sex_lable_history.pack()
            experssion_lable_history.pack()
            glass_lable_history.pack()
            hat_lable_history.pack()


##调用阿里云视觉api（检测口罩）
###################################################################################
def DetectMask(oss_url):
    mask_info = []
    request = DetectMaskRequest()
    request.set_accept_format('json')

    request.set_ImageURL(oss_url)

    response = client.do_action_with_exception(request)
    response_dict = json.loads(response)
    Mask = response_dict["Data"]["Mask"]
    # print(Mask)
    if (Mask == 1):
        Mask_text = "没有佩戴口罩！"
        color = "red"
    elif (Mask == 2):
        Mask_text = "正确佩戴口罩！"
        color = "green"
    elif (Mask == 3):
        Mask_text = "错误佩戴口罩！"
        color = "yellow"
    mask_info.append(Mask_text)
    mask_info.append(color)
    # print(Mask_text)
    return mask_info
    # print(str(response, encoding='utf-8'))


# 调用阿里云视觉API（探测人脸属性）
##################################################################################
def face_attributes(oss_url):
    attributes = []
    request = RecognizeFaceRequest()
    request.set_accept_format('json')

    request.set_ImageURL(oss_url)

    response = client.do_action_with_exception(request)
    response_dict = json.loads(response)
    print(response_dict)

    age = response_dict["Data"]["AgeList"][0]
    gender = response_dict["Data"]["GenderList"][0]
    beauty = response_dict["Data"]["BeautyList"][0]
    # experssion = response_dict["Data"]["Expressions"][0]
    glass = response_dict["Data"]["Glasses"][0]
    hat = response_dict["Data"]["HatList"][0]

    if (gender == 0):
        sex = "女"
    else:
        sex = "男"

    if (glass == 0):
        glass_text = "不戴眼镜"
    else:
        glass_text = "戴眼镜"

    if (hat == 0):
        hat_text = "无帽子"
    else:
        hat_text = "有帽子"

    attributes.append(sex)
    attributes.append(age)
    attributes.append(beauty)
    attributes.append(glass_text)
    attributes.append(hat_text)

    # print(str(response, encoding='utf-8'))
    return attributes


##调用阿里云API（探测人脸表情）
##################################################################################
def detect_experssions(oss_url):
    request = RecognizeExpressionRequest()
    request.set_accept_format('json')
    request.set_ImageURL(oss_url)

    response = client.do_action_with_exception(request)
    response_dict = json.loads(response)
    expression = response_dict["Data"]["Elements"][0]["Expression"]
    # print(expression)
    # ：neutral（中性）、happiness（高兴）、surprise（惊讶）、sadness（伤心）、anger（生气）、disgust（厌恶）、fear（害怕）。
    if (expression == "neutral"):
        expression_text = "中性"
    elif (expression == "happiness"):
        expression_text = "高兴"
    elif (expression == "surprise"):
        expression_text = "惊讶"
    elif (expression == "sadness"):
        expression_text = "伤心"
    elif (expression == "anger"):
        expression_text = "生气"
    elif (expression == "disgust"):
        expression_text = "厌恶"
    elif (expression == "fear"):
        expression_text = "害怕"

    return expression_text

    # python2:  print(response) 
    # print(str(response, encoding='utf-8'))


##################################################################################
# 实例化tkinter对象，建立窗口window
window = tk.Tk()
# 设置标题
window.title("人脸口罩、属性检测")
# 获取屏幕分辨率
screenWidth = window.winfo_screenwidth()
screenHeight = window.winfo_screenheight()
# 窗口大小
windowWidth = 1000
windowHeight = 500
x = int((screenWidth - windowWidth) / 2)
y = int((screenHeight - windowHeight) / 2)
# 设置窗口大小,在屏幕中央
window.geometry("%sx%s+%s+%s" % (windowWidth, windowHeight, x, y))

# 建立主Frame，在window上
frame = tk.Frame(window, width=windowWidth, height=windowHeight)
frame.pack_propagate(0)
frame.pack()

# 建立第二层Frame框架，在第一层Frame上
frame_l = tk.Frame(frame, width=650, height=windowHeight)
frame_l.pack_propagate(0)
frame_l.pack(side="left")
frame_r = tk.Frame(frame, width=350, height=windowHeight)
frame_r.pack_propagate(0)
frame_r.pack(side="right")
# 刷新按钮
fresh_button = tk.Button(frame_r, text="刷新", command=info_update)
fresh_button.pack(side="bottom")

# 显示信息的canvas
canvas = tk.Canvas(frame_r, width=350, height=windowHeight, scrollregion=(0, 0, 5000, 5000))
frame_c = tk.Frame(canvas)
lable_now = tk.Label(frame_c, text="当前视频中人脸", anchor="w")
frame_c_now = tk.Frame(frame_c, width=350)
lable_history = tk.Label(frame_c, text="历史记录", anchor="w")

frame_c_history = tk.Frame(frame_c, width=350)

canvas.create_window((0, 0), window=frame_c, anchor="nw")
# 滚动条
vbar = tk.Scrollbar(frame_r, orient="vertical", takefocus=0.5)
vbar.pack(side=tk.RIGHT, fill=tk.Y)
vbar.config(command=canvas.yview)

canvas.config(yscrollcommand=vbar.set)
canvas.pack(side="left", expand=True, fill="both")
canvas.pack()
# frame_c.pack()
lable_now.pack()
frame_c_now.pack()
lable_history.pack()
frame_c_history.pack()

# 显示摄像头实时画面
camera_panel = tk.Label(frame_l)
camera_panel.pack(side="left")
# 显示实时信息


# 多线程调用
# thread1 = threading.Thread(target=video_loop)
# thread2 = threading.Thread(target=video_loop2)
# thread1.start()
# thread2.start()
video_loop()
window.mainloop()
camera.release()
cv2.destroyAllWindows()
