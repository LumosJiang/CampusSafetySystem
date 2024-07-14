from datetime import datetime

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
from threading import Thread
from flask import Flask, Response

from alive.src.generate_patches import CropImage
from alive.test import process_frame
from alive.test import load_models


import mysql.connector
from mysql.connector import Error

# MySQL database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'mydatabase'
}


# Initialize Flask application
app = Flask(__name__)

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.face_features_known_list = []
        # 存储录入人脸名字 / Save the name of faces in the database
        self.face_name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数 / Reclassify after 'reclassify_interval' frames
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        self.blacklist_users = []  # 加载黑名单用户列表

        self.model_test, self.models = load_models("alive//resources/anti_spoof_models",0 )
        self.label = 0

    def log_forbidden_event(self, event_time, location, event_type):
        try:
            # 获取当前目录下已有的视频文件数量
            video_dir = r'D:\nodejsProjects\project11\Frontend\public\pictures\fobbiden'
            current_videos = [filename for filename in os.listdir(video_dir) if
                              filename.startswith('output') and filename.endswith('.jpg')]
            video_count = len(current_videos)

            # 新视频文件路径和名称
            video_name = f'output_{video_count}.jpg'
            video_path = os.path.join("../../public/pictures/fobbiden/", video_name)

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
    def log_fake_event(self, event_time, location, event_type):
        try:
            # 获取当前目录下已有的视频文件数量
            video_dir = r'D:\nodejsProjects\project11\Frontend\public\pictures\fake'
            current_videos = [filename for filename in os.listdir(video_dir) if
                              filename.startswith('output') and filename.endswith('.jpg')]
            video_count = len(current_videos)

            # 新视频文件路径和名称
            video_name = f'output_{video_count}.jpg'
            video_path = os.path.join("../../public/pictures/fake/", video_name)

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
    def load_blacklist_users(self):
        blacklist_dir = 'data/data_faces_blacklist'
        blacklist_users = []
        for folder_name in os.listdir(blacklist_dir):
            if os.path.isdir(os.path.join(blacklist_dir, folder_name)):
                blacklist_users.append(folder_name)
        print(blacklist_users)
        return blacklist_users

    def save_fake_face_image(self, frame):
        # 确保目录存在，如果不存在则创建
        fobidden_faces_dir = "D:/nodejsProjects/project11/Frontend/public/pictures/fake"
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

    def save_fobbiden_face_image(self, frame):
        # 确保目录存在，如果不存在则创建
        fobidden_faces_dir = "D:/nodejsProjects/project11/Frontend/public/pictures/fobbiden"
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
    def get_next_video_filename(self):
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
            cursor.execute(insert_query, (next_video_id, video_path, video_datetime, video_name, "活体检测"))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error inserting video info into database: {e}")

        return video_path

    def load_blacklist_users(self):
        blacklist_dir = 'data/data_faces_blacklist'
        blacklist_users = []
        for folder_name in os.listdir(blacklist_dir):
            if os.path.isdir(os.path.join(blacklist_dir, folder_name)):
                blacklist_users.append(folder_name)
        print(blacklist_users)
        return blacklist_users

    def get_frame(self, stream):
        self.blacklist_users_logs = []
        self.blacklist_users = self.load_blacklist_users()

        cap = cv2.VideoCapture('rtmp://101.200.135.114:9090/live/live1')

        # 获取视频帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置视频编码和输出文件
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out_filename = self.get_next_video_filename()
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()

                # 2. 检测人脸 / Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3. 更新人脸计数器 / Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4. 更新上一帧中的人脸列表 / Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. 更新上一帧和当前帧的质心列表 / update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 如果当前帧和上一帧人脸数没有变化 / if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug(
                        "scene 1: 当前帧和上一帧相比没有发生人脸数变化 / No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        logging.debug("  有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                        self.reclassify_interval_cnt += 1
                        self.log_forbidden_event(event_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                 location="教室",
                                                 event_type="未知人员")
                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    # 如果当前帧中有多个人脸, 使用质心追踪 / Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # Check if the current face feature is in the blacklist
                        if self.current_frame_face_name_list[i] in self.blacklist_users:
                            if self.current_frame_face_name_list[i] not in self.blacklist_users_logs:
                                # 记录异常事件
                                self.log_forbidden_event(event_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                         location="教室",
                                                         event_type="黑名单人员")
                                self.blacklist_users_logs.append(self.current_frame_face_name_list[i])
                            img_rd = cv2.putText(img_rd, "Forbidden",
                                                 self.current_frame_face_position_list[i], self.font, 0.8, (0, 0, 255),
                                                 1, cv2.LINE_AA)
                            self.save_fobbiden_face_image(img_rd)
                        else:
                            self.label = process_frame(img_rd, self.model_test, CropImage(),self.models)
                            if self.label == 1:
                                img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                                     self.current_frame_face_position_list[i], self.font, 0.8,
                                                     (0, 255, 255), 1, cv2.LINE_AA)
                            else:
                                img_rd = cv2.putText(img_rd, "Fake",
                                                     self.current_frame_face_position_list[i], self.font, 0.8,
                                                     (0, 255, 0), 1, cv2.LINE_AA)
                                # 记录异常事件
                                self.log_fake_event(event_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                         location="教室",
                                                         event_type="非活体入侵")
                                self.save_fake_face_image(img_rd)

                    self.draw_note(img_rd)

                # 6.2 如果当前帧和上一帧人脸数发生变化 / If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1 人脸数减少 / Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  scene 2.1 人脸消失, 当前帧中没有人脸 / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 人脸数增加 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug(
                            "  scene 2.2 出现人脸, 进行人脸识别 / Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                # 如果 q 数据不为空
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                if self.current_frame_face_name_list[k] in self.blacklist_users:
                                    self.current_frame_face_name_list[k] = "Forbidden"
                                    logging.debug("  Face recognition result: Forbidden (blacklisted)")
                                else:
                                    self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                    logging.debug("  Face recognition result: %s",
                                                  self.face_name_known_list[similar_person_num])
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                self.update_fps()
                # cv2.namedWindow("camera", 1)
                # cv2.imshow("camera", img_rd)

                # Your existing frame processing code here

                # 写入视频文件
                out.write(img_rd)
                # 处理后的视频帧转换为JPEG格式
                ret, jpeg = cv2.imencode('.jpg', img_rd)
                frame = jpeg.tobytes()

                # 返回multipart/x-mixed-replace格式的视频流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cap.release()
            out.release()
    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 使用质心追踪来识别人脸 / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算 / For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]


    # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some info on windows
       # cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "Face Register", (20, 150), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition", (20, 270), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 380), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Detected faces: %d" % self.current_frame_face_cnt, (20, 500), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 620), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 进行人脸识别 / Do face recognition
face_recognizer = Face_Recognizer()

@app.route('/')
def index():
    return "Default Page of Flask Server"


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognizer.get_frame(cv2.VideoCapture("rtmp://101.200.135.114:9090/live/live1")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5004)



