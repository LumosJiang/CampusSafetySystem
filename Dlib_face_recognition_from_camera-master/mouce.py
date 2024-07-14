from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np


def MAR(mouth):
    # 默认二范数：求特征值，然后求最大特征值得算术平方根
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59（人脸68个关键点）
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55

    return (A + B) / (2.0 * C)


def main():
    MAR_THRESH = 0.5  # 张嘴阈值

    # 初始化
    COUNTER_MOUTH = 0
    TOTAL_MOUTH = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')

    # 嘴的索引
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度框中检测人脸
        rects = detector(gray, 0)

        # 进入循环
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 提取嘴唇坐标，然后使用该坐标计算嘴唇纵横比
            Mouth = shape[mStart:mEnd]
            mar = MAR(Mouth)
            # 判断嘴唇纵横比是否高于张嘴阈值，如果是，则增加张嘴帧计数器
            if mar > MAR_THRESH:
                COUNTER_MOUTH += 1

            else:
                # 如果张嘴帧计数器不等于0，则增加张嘴的总次数
                if COUNTER_MOUTH >= 2:
                    TOTAL_MOUTH += 1
                COUNTER_MOUTH = 0

            cv2.putText(frame, "Mouth is open: {}".format(TOTAL_MOUTH), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()