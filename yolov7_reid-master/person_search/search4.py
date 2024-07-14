# import argparse
# import random
# import torch
# import torch.nn.functional as F
# import os
# import sys
#
# from torch.backends import cudnn
#
# from models.experimental import attempt_load
#
# sys.path.append('.')
# from reid.data.transforms import build_transforms
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, non_max_suppression, increment_path,scale_coords
# from utils.plots import plot_one_box
# from reid.data import make_data_loader
# from pathlib import Path
# from reid.modeling import build_model
# from reid.config import cfg as reidCfg
# import numpy as np
# from PIL import Image
# import cv2
# from loguru import logger
#
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv7 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#
# def detect(
#            opt,
#            source=['0'],
#            imgsz=(640, 640),
#            weights='yolov7.pt',
#            half=False,
#            dist_thres=1.0,
#            save_res=False,
#            project='runs/detect',
#            name='exp',
#            exist_ok=False,
# ):
#     source = str(source)
#     save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
#     save_dir.mkdir(parents=True, exist_ok=True)  # make dir
#     device = torch.device('cpu')
#     torch.backends.cudnn.benchmark = False  # set False for reproducible results
#
#     # ---------- 行人重识别模型初始化 --------------------------
#     reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
#     reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
#     reidModel.to(device).eval()  # 模型测试
#
#     # --------------- yolov7 行人检测模型初始化 -------------------
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size
#     # Dataloader
#     cudnn.benchmark = True  # set True to speed up constant image size inference
#     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#
#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#
#     # Run inference
#     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         # Inference
#         pred = model(img, augment=False)[0]
#         # NMS
#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
#         # Process predictions
#
#         gallery_loc = []  # 这个列表用来存放框的坐标
#         # 假设一个字典来存储已识别的行人特征和对应的ID
#         identified_people = {}
#         next_id = 0
#         for i, det in enumerate(pred):  # detections per image
#             im0, frame = im0s[i].copy(), dataset.count
#             current_frame_ids = set()  # 用于存储当前帧中已分配的ID
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 for *xyxy, conf, cls in reversed(det):
#                     if names[int(cls)] == 'person':
#                         xmin = int(xyxy[0])
#                         ymin = int(xyxy[1])
#                         xmax = int(xyxy[2])
#                         ymax = int(xyxy[3])
#                         gallery_loc.append((xmin, ymin, xmax, ymax))
#                         crop_img = im0[ymin:ymax, xmin:xmax]  # 获取该帧中框的位置
#                         crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
#                         crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # 应用变换
#                         gallery_img = crop_img.to(device)
#                         gallery_feats = reidModel(gallery_img)  # 计算特征向量
#                         gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 归一化特征向量
#                         # 比较当前行人的特征与已识别的行人特征
#                         found_match = False
#                         matched_id = None
#                         min_dist = dist_thres
#                         for id, feats_list in identified_people.items():
#                             # 计算当前行人与已知行人的特征之间的距离
#                             for feat in feats_list:
#                                 m, n = feat.shape[0], gallery_feats.shape[0]
#                                 distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                                           torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#
#                                 distmat.addmm_(1, -2, feat, gallery_feats.t())
#                                 distmat = distmat.cpu().numpy()
#                                 distmat = distmat.sum(axis=0) / len(feat)  # 平均一下query中同一行人的多个结果
#                                 index = distmat.argmin()
#                                 if distmat[index] < min_dist:  # 如果距离小于阈值，认为是同一个人
#                                     # 使用已知的ID
#                                     found_match = True
#                                     matched_id = id
#                                     min_dist = distmat[index]
#
#
#                         # 如果当前行人没有匹配到已知的ID，分配一个新的ID
#                         if not found_match:
#                             next_id += 1
#                             matched_id = next_id
#                             identified_people[matched_id] = [gallery_feats]
#                         else:
#                             match_in = True
#                             while matched_id in current_frame_ids:  # 确保ID是唯一的
#                                 next_id += 1
#                                 matched_id = next_id
#                                 identified_people[matched_id] = [gallery_feats]
#                                 match_in = False
#                                 break
#                             if match_in:
#                                 identified_people[matched_id].append(gallery_feats)
#
#
#                         current_frame_ids.add(matched_id)
#                         plot_one_box(gallery_loc[-1], im0, label=f'person {matched_id}', color=colors_[int(cls)])
#
#             torch.cuda.empty_cache()
#             cv2.imshow(f'person search{i}', im0)
#             cv2.waitKey(2)
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='person search')
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path or triton URL')
#     parser.add_argument('--source', type=str, default="source.txt", help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=480,
#                         help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
#     parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')
#
#     opt = parser.parse_args()
#     logger.info(opt)
#     weights, source, imgsz, half, dist_thres, save_res = opt.weights, opt.source, opt.imgsz,  opt.half, opt.dist_thres, opt.save_res
#
#     with torch.no_grad():
#         detect(opt, source, imgsz, weights, half, dist_thres=dist_thres, save_res=save_res)
# #
#
#
#
#
# #
# # import argparse
# # import random
# # import torch
# # import torch.nn.functional as F
# # import os
# # import sys
# #
# # from flask import app, Response, Flask
# # from torch.backends import cudnn
# #
# # from models.experimental import attempt_load
# #
# # sys.path.append('.')
# # from reid.data.transforms import build_transforms
# # from utils.datasets import LoadStreams, LoadImages
# # from utils.general import check_img_size, non_max_suppression, increment_path,scale_coords
# # from utils.plots import plot_one_box
# # from reid.data import make_data_loader
# # from pathlib import Path
# # from reid.modeling import build_model
# # from reid.config import cfg as reidCfg
# # import numpy as np
# # from PIL import Image
# # import cv2
# # from loguru import logger
# #
# # app = Flask(__name__)
# #
# # FILE = Path(__file__).resolve()
# # ROOT = FILE.parents[0]  # YOLOv7 root directory
# # if str(ROOT) not in sys.path:
# #     sys.path.append(str(ROOT))  # add ROOT to PATH
# # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# #
# # def detect(
# #            opt,
# #            source=['0'],
# #            imgsz=(640, 640),
# #            weights='yolov7.pt',
# #            half=False,
# #            dist_thres=1.0,
# #            save_res=False,
# #            project='runs/detect',
# #            name='exp',
# #            exist_ok=False,
# # ):
# #     images = []
# #     source = str(source)
# #     save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
# #     save_dir.mkdir(parents=True, exist_ok=True)  # make dir
# #     device = torch.device('cpu')
# #     torch.backends.cudnn.benchmark = False  # set False for reproducible results
# #
# #     # ---------- 行人重识别模型初始化 --------------------------
# #     reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
# #     reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
# #     reidModel.to(device).eval()  # 模型测试
# #
# #     # --------------- yolov7 行人检测模型初始化 -------------------
# #     model = attempt_load(weights, map_location=device)  # load FP32 model
# #     stride = int(model.stride.max())  # model stride
# #     imgsz = check_img_size(imgsz, s=stride)  # check img_size
# #     # Dataloader
# #     cudnn.benchmark = True  # set True to speed up constant image size inference
# #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
# #
# #     # Get names and colors
# #     names = model.module.names if hasattr(model, 'module') else model.names
# #     colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# #
# #     # Run inference
# #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
# #
# #     for path, img, im0s, vid_cap in dataset:
# #         img = torch.from_numpy(img).to(device)
# #         img = img.half() if half else img.float()  # uint8 to fp16/32
# #         img /= 255.0  # 0 - 255 to 0.0 - 1.0
# #         if img.ndimension() == 3:
# #             img = img.unsqueeze(0)
# #         # Inference
# #         pred = model(img, augment=False)[0]
# #         # NMS
# #         # Apply NMS
# #         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
# #         # Process predictions
# #
# #         gallery_loc = []  # 这个列表用来存放框的坐标
# #         # 假设一个字典来存储已识别的行人特征和对应的ID
# #         identified_people = {}
# #         next_id = 0
# #         for i, det in enumerate(pred):  # detections per image
# #             im0, frame = im0s[i].copy(), dataset.count
# #             current_frame_ids = set()  # 用于存储当前帧中已分配的ID
# #             if len(det):
# #                 # Rescale boxes from img_size to im0 size
# #                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
# #
# #                 for *xyxy, conf, cls in reversed(det):
# #                     if names[int(cls)] == 'person':
# #                         xmin = int(xyxy[0])
# #                         ymin = int(xyxy[1])
# #                         xmax = int(xyxy[2])
# #                         ymax = int(xyxy[3])
# #                         gallery_loc.append((xmin, ymin, xmax, ymax))
# #                         crop_img = im0[ymin:ymax, xmin:xmax]  # 获取该帧中框的位置
# #                         crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
# #                         crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # 应用变换
# #                         gallery_img = crop_img.to(device)
# #                         gallery_feats = reidModel(gallery_img)  # 计算特征向量
# #                         gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 归一化特征向量
# #                         # 比较当前行人的特征与已识别的行人特征
# #                         found_match = False
# #                         matched_id = None
# #                         min_dist = dist_thres
# #                         for id, feats_list in identified_people.items():
# #                             # 计算当前行人与已知行人的特征之间的距离
# #                             for feat in feats_list:
# #                                 m, n = feat.shape[0], gallery_feats.shape[0]
# #                                 distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
# #                                           torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
# #
# #                                 distmat.addmm_(1, -2, feat, gallery_feats.t())
# #                                 distmat = distmat.detach().cpu().numpy()
# #                                 distmat = distmat.sum(axis=0) / len(feat)  # 平均一下query中同一行人的多个结果
# #                                 index = distmat.argmin()
# #                                 if distmat[index] < min_dist:  # 如果距离小于阈值，认为是同一个人
# #                                     # 使用已知的ID
# #                                     found_match = True
# #                                     matched_id = id
# #                                     min_dist = distmat[index]
# #
# #
# #                         # 如果当前行人没有匹配到已知的ID，分配一个新的ID
# #                         if not found_match:
# #                             next_id += 1
# #                             matched_id = next_id
# #                             identified_people[matched_id] = [gallery_feats]
# #                         else:
# #                             match_in = True
# #                             while matched_id in current_frame_ids:  # 确保ID是唯一的
# #                                 next_id += 1
# #                                 matched_id = next_id
# #                                 identified_people[matched_id] = [gallery_feats]
# #                                 match_in = False
# #                                 break
# #                             if match_in:
# #                                 identified_people[matched_id].append(gallery_feats)
# #
# #
# #                         current_frame_ids.add(matched_id)
# #                         plot_one_box(gallery_loc[-1], im0, label=f'person {matched_id}', color=colors_[int(cls)])
# #             images.append(im0)
# #             torch.cuda.empty_cache()
# #             # 处理后的视频帧转换为JPEG格式
# #             ret, jpeg = cv2.imencode('.jpg', im0)
# #             frame = jpeg.tobytes()
# #
# #             # 返回multipart/x-mixed-replace格式的视频流
# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# #
# #
# # @app.route('/video_feed')
# # def video_feed():
# #     parser = argparse.ArgumentParser(description='person search')
# #     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path or triton URL')
# #     parser.add_argument('--source', type=str, default="source.txt", help='file/dir/URL/glob/screen/0(webcam)')
# #     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=480,
# #                         help='inference size h,w')
# #     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
# #     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# #     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# #     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
# #     parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
# #     parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')
# #
# #     opt = parser.parse_args()
# #     logger.info(opt)
# #     weights, source, imgsz, half, dist_thres, save_res = opt.weights, opt.source, opt.imgsz, opt.half, opt.dist_thres, opt.save_res
# #
# #     # Video streaming route. Put this in the src attribute of an img tag
# #     return Response(detect(opt, source, imgsz, weights, half, dist_thres=dist_thres, save_res=save_res),
# #                     mimetype='multipart/x-mixed-replace; boundary=frame')
# #
# #
# # @app.route('/')
# # def index():
# #     # Video streaming route. Put this in the src attribute of an img tag
# #      return "render_template('index.html')"
# #
# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5006, debug=True)
#
#
#
#
import argparse
import random
import torch
import torch.nn.functional as F
import os
import sys

from flask import app, Response, Flask
from torch.backends import cudnn

from models.experimental import attempt_load

sys.path.append('.')
from reid.data.transforms import build_transforms
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, increment_path,scale_coords
from utils.plots import plot_one_box
from reid.data import make_data_loader
from pathlib import Path
from reid.modeling import build_model
from reid.config import cfg as reidCfg
import numpy as np
from PIL import Image
import cv2
from loguru import logger

app = Flask(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# def detect(
#            opt,
#            source=['0'],
#            imgsz=(640, 640),
#            weights='yolov7.pt',
#            half=False,
#            dist_thres=1.0,
#            save_res=False,
#            project='runs/detect',
#            name='exp',
#            exist_ok=False,
# ):
#     images = []
#     source = str(source)
#     save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
#     save_dir.mkdir(parents=True, exist_ok=True)  # make dir
#     device = torch.device('cpu')
#     torch.backends.cudnn.benchmark = False  # set False for reproducible results
#
#     # ---------- 行人重识别模型初始化 --------------------------
#     reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
#     reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
#     reidModel.to(device).eval()  # 模型测试
#
#     # --------------- yolov7 行人检测模型初始化 -------------------
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size
#     # Dataloader
#     cudnn.benchmark = True  # set True to speed up constant image size inference
#     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#
#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#
#     # Run inference
#     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         # Inference
#         pred = model(img, augment=False)[0]
#         # NMS
#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
#         # Process predictions
#
#         gallery_loc = []  # 这个列表用来存放框的坐标
#         # 假设一个字典来存储已识别的行人特征和对应的ID
#         identified_people = {}
#         next_id = 0
#         for i, det in enumerate(pred):  # detections per image
#             im0, frame = im0s[i].copy(), dataset.count
#             current_frame_ids = set()  # 用于存储当前帧中已分配的ID
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 for *xyxy, conf, cls in reversed(det):
#                     if names[int(cls)] == 'person':
#                         xmin = int(xyxy[0])
#                         ymin = int(xyxy[1])
#                         xmax = int(xyxy[2])
#                         ymax = int(xyxy[3])
#                         gallery_loc.append((xmin, ymin, xmax, ymax))
#                         crop_img = im0[ymin:ymax, xmin:xmax]  # 获取该帧中框的位置
#                         crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
#                         crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # 应用变换
#                         gallery_img = crop_img.to(device)
#                         gallery_feats = reidModel(gallery_img)  # 计算特征向量
#                         gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 归一化特征向量
#                         # 比较当前行人的特征与已识别的行人特征
#                         found_match = False
#                         matched_id = None
#                         min_dist = dist_thres
#                         for id, feats_list in identified_people.items():
#                             # 计算当前行人与已知行人的特征之间的距离
#                             for feat in feats_list:
#                                 m, n = feat.shape[0], gallery_feats.shape[0]
#                                 distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                                           torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#
#                                 distmat.addmm_(1, -2, feat, gallery_feats.t())
#                                 distmat = distmat.detach().cpu().numpy()
#                                 distmat = distmat.sum(axis=0) / len(feat)  # 平均一下query中同一行人的多个结果
#                                 index = distmat.argmin()
#                                 if distmat[index] < min_dist:  # 如果距离小于阈值，认为是同一个人
#                                     # 使用已知的ID
#                                     found_match = True
#                                     matched_id = id
#                                     min_dist = distmat[index]
#
#
#                         # 如果当前行人没有匹配到已知的ID，分配一个新的ID
#                         if not found_match:
#                             next_id += 1
#                             matched_id = next_id
#                             identified_people[matched_id] = [gallery_feats]
#                         else:
#                             match_in = True
#                             while matched_id in current_frame_ids:  # 确保ID是唯一的
#                                 next_id += 1
#                                 matched_id = next_id
#                                 identified_people[matched_id] = [gallery_feats]
#                                 match_in = False
#                                 break
#                             if match_in:
#                                 identified_people[matched_id].append(gallery_feats)
#
#
#                         current_frame_ids.add(matched_id)
#                         plot_one_box(gallery_loc[-1], im0, label=f'person {matched_id}', color=colors_[int(cls)])
#             images.append(im0)
#             torch.cuda.empty_cache()
#             # 处理后的视频帧转换为JPEG格式
#             ret, jpeg = cv2.imencode('.jpg', im0)
#             frame = jpeg.tobytes()
#
#             # 返回multipart/x-mixed-replace格式的视频流
#             yield (b'--frame\r\n'
#                     b'Content-Type: text/plain\r\n\r\n' + str(i).encode() + b'\r\n' +
#                     b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
# # 在请求处理之前执行的方法
#
# # 处理视频流的视图函数
# @app.route('/video_feed/<int:stream_id>')
# def video_feed(stream_id):
#     parser = argparse.ArgumentParser(description='person search')
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path or triton URL')
#     parser.add_argument('--source', type=str, default="source.txt", help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=480,
#                         help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
#     parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')
#
#     opt = parser.parse_args()
#     logger.info(opt)
#     weights, source, imgsz, half, dist_thres, save_res = opt.weights, opt.source, opt.imgsz, opt.half, opt.dist_thres, opt.save_res
#
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(detect(opt, source, imgsz, weights, half, dist_thres=dist_thres, save_res=save_res) ,
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# @app.route('/')
# def index():
#     # Video streaming route. Put this in the src attribute of an img tag
#      return "render_template('index.html')"
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5006, debug=True)







def detect(
        opt,
        source=['0'],
        imgsz=(640, 640),
        weights='yolov7.pt',
        half=False,
        dist_thres=1.0,
        save_res=False,
        project='runs/detect',
        name='exp',
        exist_ok=False,
        stream_id = None
):
    # 每个线程使用独立的 images 列表
    thread_images = []
    source = str(source)
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    device = torch.device('cpu')
    torch.backends.cudnn.benchmark = False  # set False for reproducible results

    # 行人重识别模型初始化
    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
    reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
    reidModel.to(device).eval()  # 模型测试

    # yolov7 行人检测模型初始化
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # 获取类名和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)

        gallery_loc = []  # 这个列表用来存放框的坐标
        identified_people = {}
        next_id = 0
        for i, det in enumerate(pred):  # detections per image
            im0, frame = im0s[i].copy(), dataset.count
            current_frame_ids = set()  # 用于存储当前帧中已分配的ID
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == 'person':
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax, xmin:xmax]  # 获取该帧中框的位置
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
                        crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # 应用变换
                        gallery_img = crop_img.to(device)
                        gallery_feats = reidModel(gallery_img)  # 计算特征向量
                        gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 归一化特征向量

                        # 比较当前行人的特征与已识别的行人特征
                        found_match = False
                        matched_id = None
                        min_dist = dist_thres
                        for id, feats_list in identified_people.items():
                            for feat in feats_list:
                                m, n = feat.shape[0], gallery_feats.shape[0]
                                distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                          torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                                distmat.addmm_(1, -2, feat, gallery_feats.t())
                                distmat = distmat.detach().cpu().numpy()
                                distmat = distmat.sum(axis=0) / len(feat)  # 平均一下query中同一行人的多个结果
                                index = distmat.argmin()
                                if distmat[index] < min_dist:  # 如果距离小于阈值，认为是同一个人
                                    found_match = True
                                    matched_id = id
                                    min_dist = distmat[index]

                        if not found_match:
                            next_id += 1
                            matched_id = next_id
                            identified_people[matched_id] = [gallery_feats]
                        else:
                            while matched_id in current_frame_ids:  # 确保ID是唯一的
                                next_id += 1
                                matched_id = next_id
                                identified_people[matched_id] = [gallery_feats]
                            identified_people[matched_id].append(gallery_feats)

                        current_frame_ids.add(matched_id)
                        plot_one_box(gallery_loc[-1], im0, label=f'person {matched_id}', color=colors_[int(cls)])

            thread_images.append(im0)  # 只保存当前线程的图像
            torch.cuda.empty_cache()
            # 处理后的视频帧转换为JPEG格式
            ret, jpeg = cv2.imencode('.jpg', im0)
            frame = jpeg.tobytes()
            if stream_id == i:
                # 返回multipart/x-mixed-replace格式的视频流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 处理视频流的视图函数
@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    parser = argparse.ArgumentParser(description='person search')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default="source.txt", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=480,
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
    parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')

    opt = parser.parse_args()
    logger.info(opt)
    weights, source, imgsz, half, dist_thres, save_res = opt.weights, opt.source, opt.imgsz, opt.half, opt.dist_thres, opt.save_res

    # Video streaming route. Put this in the src attribute of an img tag
    return Response(detect(opt, source, imgsz, weights, half, dist_thres=dist_thres, save_res=save_res,stream_id=stream_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008)
