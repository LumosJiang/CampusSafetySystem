from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2


def nose_jaw_distance(nose, jaw):
    # 计算鼻子上一点"27"到左右脸边界的欧式距离
    face_left1 = dist.euclidean(nose[0], jaw[0])  # 27, 0
    face_right1 = dist.euclidean(nose[0], jaw[16])  # 27, 16
    # 计算鼻子上一点"30"到左右脸边界的欧式距离
    face_left2 = dist.euclidean(nose[3], jaw[2])  # 30, 2
    face_right2 = dist.euclidean(nose[3], jaw[14])  # 30, 14
    # 创建元组，用以保存4个欧式距离值
    face_distance = (face_left1, face_right1, face_left2, face_right2)

    return face_distance


def main():
    # 初始化眨眼帧计数器和总眨眼次数
    distance_left = 0
    distance_right = 0
    TOTAL_FACE = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']

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

            # 提取鼻子和下巴的坐标，然后使用该坐标计算鼻子到左右脸边界的欧式距离
            nose = shape[nStart:nEnd]
            jaw = shape[jStart:jEnd]
            NOSE_JAW_Distance = nose_jaw_distance(nose, jaw)
            # 移植鼻子到左右脸边界的欧式距离
            face_left1 = NOSE_JAW_Distance[0]
            face_right1 = NOSE_JAW_Distance[1]
            face_left2 = NOSE_JAW_Distance[2]
            face_right2 = NOSE_JAW_Distance[3]

            # 根据鼻子到左右脸边界的欧式距离，判断是否摇头
            # 左脸大于右脸
            if face_left1 >= face_right1 + 2 and face_left2 >= face_right2 + 2:
                distance_left += 1
            # 右脸大于左脸
            if face_right1 >= face_left1 + 2 and face_right2 >= face_left2 + 2:
                distance_right += 1
            # 左脸大于右脸，并且右脸大于左脸，判定摇头
            if distance_left != 0 and distance_right != 0:
                TOTAL_FACE += 1
                distance_right = 0
                distance_left = 0

            # 画出摇头次数
            cv2.putText(frame, "shake one's head: {}".format(TOTAL_FACE), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()