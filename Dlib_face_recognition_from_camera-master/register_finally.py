import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging

from flask import Flask, render_template, Response, request, jsonify

import features_extraction_to_csv

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()

class FaceRegister:
    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.current_frame_faces_cnt = 0

        self.save_flag = 1
        self.press_n_flag = 0

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    def pre_work_mkdir(self):
        if not os.path.isdir(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            person_list = os.listdir("data/data_faces_from_camera/")
            # person_num_list = [int(person.split('_')[-1]) for person in person_list]
            # self.existing_faces_cnt = max(person_num_list)
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)
        else:
            self.existing_faces_cnt = 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps_show.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        faces = detector(frame, 0)
        for k, d in enumerate(faces):
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height/2)
            ww = int(width/2)

            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(frame, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                self.save_flag = 0
            else:
                color_rectangle = (255, 255, 255)
                self.save_flag = 1

            cv2.rectangle(frame, tuple([d.left() - ww, d.top() - hh]), tuple([d.right() + ww, d.bottom() + hh]), color_rectangle, 2)

            if self.save_flag and self.press_n_flag:
                img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)
                for ii in range(height*2):
                    for jj in range(width*2):
                        img_blank[ii][jj] = frame[d.top()-hh + ii][d.left()-ww + jj]
                cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                self.ss_cnt += 1
                self.save_flag = 0
                self.press_n_flag = 0

        self.current_frame_faces_cnt = len(faces)
        self.draw_note(frame)
        self.update_fps()
        return frame

    def register_face(self, action, username):
        self.save_flag = 0
        self.press_n_flag = 0
        if action == 'n':
            self.existing_faces_cnt += 1
            self.current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)+"_"+username
            os.makedirs(self.current_face_dir)
            self.ss_cnt = 0
            self.press_n_flag = 1
        elif action == 's' and self.press_n_flag:
            self.save_flag = 1

face_register = FaceRegister()
face_register.pre_work_mkdir()
face_register.check_existing_faces_cnt()

@app.route('/')
def index():
    return "hello"

def gen():
    cap = cv2.VideoCapture("rtmp://101.200.135.114:9090/live/live1")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = face_register.process_frame(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    action = request.form.get('action')
    username = request.form.get('username')  # 获取用户名
    face_register.register_face(action, username)

    # 等待两秒钟
    time.sleep(2)

    features_extraction_to_csv.main()
    return jsonify({'status': 'success', 'action': action})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
