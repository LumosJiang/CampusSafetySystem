# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import imutils
# import time
# import dlib
# import cv2
#
#
# def EAR(eye):
#     # 计算眼睛的两组垂直关键点之间的欧式距离
#     A = dist.euclidean(eye[1], eye[5])  # 1,5是一组垂直关键点
#     B = dist.euclidean(eye[2], eye[4])  # 2,4是一组
#     # 计算眼睛的一组水平关键点之间的欧式距离
#     C = dist.euclidean(eye[0], eye[3])  # 0,3是一组水平关键点
#
#     return (A + B) / (2.0 * C)
#
#
# def eye_detect_main():
#     EAR_THRESH = 0.3  # 眨眼阈值
#     EYE_close = 2  # 闭眼次数阈值
#
#     # 初始化眨眼帧计数器和总眨眼次数
#     count_eye = 0
#     total = 0
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
#
#     # 左右眼的索引
#     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)
#
#     while True:
#
#         frame = vs.read()
#         frame = imutils.resize(frame, width=600)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # 在灰度框中检测人脸
#         rects = detector(gray, 0)
#
#         # 进入循环
#         for rect in rects:
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#
#             # 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
#             leftEye = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             ear = EAR(leftEye) + EAR(rightEye) / 2.0
#             # 判断眼睛纵横比是否低于眨眼阈值
#             if ear < EAR_THRESH:
#                 count_eye += 1
#             else:
#                 # 检测到一次闭眼
#                 if count_eye >= EYE_close:
#                     total += 1
#                 count_eye = 0
#
#             # 画出画框上眨眼的总次数以及计算出的帧的眼睛纵横比
#             cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord("q"):
#             break
#
#     cv2.destroyAllWindows()
#     vs.stop()
#
#
# if __name__ == '__main__':
#     eye_detect_main()


from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import dlib
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

EAR_THRESH = 0.3  # 眨眼阈值
EYE_close = 2  # 闭眼次数阈值

count_eye = 0
total = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 左右眼的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])  # 1,5是一组垂直关键点
    B = dist.euclidean(eye[2], eye[4])  # 2,4是一组
    C = dist.euclidean(eye[0], eye[3])  # 0,3是一组水平关键点

    return (A + B) / (2.0 * C)

def eye_detect_gen(stream):
    global count_eye, total
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (EAR(leftEye) + EAR(rightEye)) / 2.0

            if ear < EAR_THRESH:
                count_eye += 1
            else:
                if count_eye >= EYE_close:
                    total += 1
                count_eye = 0

            cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    stream = cv2.VideoCapture("rtmp://101.200.135.114:9090/live/live1")
    return Response(eye_detect_gen(stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 启动 Flask 服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
