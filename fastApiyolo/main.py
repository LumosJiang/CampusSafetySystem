import logging
from random import randint
from datetime import datetime, time
from flask import Flask, Response, render_template
import cv2
import torch
import sys
import os
import mysql.connector
from mysql.connector import Error
import time
# 确保可以导入 yolov5 文件夹中的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes, set_logging
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import letterbox

# MySQL数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'mydatabase'
}

app = Flask(__name__)

# 设置设备
device = select_device('0' if torch.cuda.is_available() else 'cpu')
half = device.type != 'cpu'  # 半精度仅支持 CUDA

# 加载训练好的 YOLOv5 模型
weights = os.path.join('yolov5', 'best.pt')
set_logging()
model = attempt_load(weights, device=device)  # 加载 FP32 模型
if half:
    model.half()  # 转换为半精度

# 获取模型的类别名
names = model.module.names if hasattr(model, 'module') else model.names

def preprocess_image(image):
    img = letterbox(image, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1].copy()  # HWC to CHW, BGR to RGB
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 绘制检测框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_next_video_filename():
    # 获取当前目录下已有的视频文件数量
    video_dir = r'D:\nodejsProjects\project11\Frontend\public\videos'
    current_videos = [filename for filename in os.listdir(video_dir) if
                      filename.startswith('output') and filename.endswith('.mp4')]
    video_count = len(current_videos)

    # 新视频文件路径和名称
    video_name = f'output_{video_count}.mp4'
    video_path = os.path.join(video_dir, video_name)

    # 获取当前时间作为视频日期
    video_datetime = datetime.now()

    # 将视频信息插入到数据库
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 获取下一个可用的 video_id
        cursor.execute("SELECT MAX(video_id) FROM Video")
        result = cursor.fetchone()
        max_id = result[0] if result[0] else 0
        next_video_id = max_id + 1

        insert_query = "INSERT INTO Video (video_id, path, date, name, type) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (next_video_id, video_path, video_datetime, video_name, "异常行为检测"))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting video info into database: {e}")

    return video_path

def save_special_face_image(frame):
    # 确保目录存在，如果不存在则创建
    fobidden_faces_dir = "D:/nodejsProjects/project11/Frontend/public/pictures/special"
    if not os.path.exists(fobidden_faces_dir):
        os.makedirs(fobidden_faces_dir)

    current_videos = [filename for filename in os.listdir(fobidden_faces_dir) if
                      filename.startswith('output') and filename.endswith('.jpg')]
    video_count = len(current_videos)

    filename = f'output_{video_count}.jpg'
    filepath = os.path.join(fobidden_faces_dir, filename)

    # 保存当前帧到文件
    cv2.imwrite(filepath, frame)
    logging.debug(f"Saved unknown face image to {filepath}")

def log_special_event(event_time, location, event_type):
    try:
        # 获取当前目录下已有的视频文件数量
        video_dir = r'D:\nodejsProjects\project11\Frontend\public\pictures\special'
        current_videos = [filename for filename in os.listdir(video_dir) if
                          filename.startswith('output') and filename.endswith('.jpg')]
        video_count = len(current_videos)

        # 新视频文件路径和名称
        video_name = f'output_{video_count}.jpg'
        video_path = os.path.join("../../public/pictures/special/", video_name)

        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = """INSERT INTO 异常事件列表 (时间, 地点, 事件,address) 
                                     VALUES (%s, %s, %s, %s)"""
            cursor.execute(insert_query, (event_time, location, event_type, video_path))
            connection.commit()
            cursor.close()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            connection.close()
def get_frame():
    # 打开本地摄像头
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('rtmp://101.200.135.114:9090/live/live1')

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频编码和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_filename = get_next_video_filename()
    out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理图像
        img = preprocess_image(frame)

        # 推理
        pred = model(img, augment=False, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # 解析结果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
                    save_special_face_image(frame)
                    log_special_event(event_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                 location="教室",
                                                 event_type=label)

        # 写入视频文件
        out.write(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()


@app.route('/')
def index():
    return "render_template('index.html')"

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)

