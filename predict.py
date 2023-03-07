import os
import cv2
import argparse
import insightface
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='insightface')
parser.add_argument('--image_path', default='image_predict', type=str, help='|要预测的图片文件夹位置|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置(.csv)|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--threshold', default=0.5, type=float, help='|概率大于阈值判断有此人|')
parser.add_argument('--device', default='cuda', type=str, help='|使用的设备cpu/cuda|')
parser.add_argument('--float16', default=False, type=bool, help='|要与特征数据库精度一致，True为float16，False为float32|')
parser.add_argument('--camera', default=True, type=bool, help='|True为启用摄像头，False为预测图片文件夹|')
parser.add_argument('--camera_time', default=20, type=int, help='|预测间隙，单位毫秒，越短显示越不卡顿但越耗性能|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def draw(image, bbox, name, color):  # 画人脸框
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
    cv2.putText(image, name, (x_min + 5, y_min + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def predict():
    # 加载模型
    model = insightface.app.FaceAnalysis(name='buffalo_l')  # 加载模型，首次运行时会自动下载模型文件到用户下的.insightface文件夹中
    model.prepare(ctx_id=-1 if args.device == 'cpu' else 0, det_size=(args.input_size, args.input_size))  # 模型设置
    # 加载数据库
    df_database = pd.read_csv(args.database_path, dtype=np.float16 if args.float16 else np.float32)
    column = df_database.columns
    feature = df_database.values
    # 开始预测
    image_dir = os.listdir(args.image_path)  # 获取预测图片路径
    for i in range(len(image_dir)):
        image = cv2.imread(args.image_path + '/' + image_dir[i])
        pred = model.get(image)
        if pred == []:
            print('| {}:未检测到人脸 |'.format(image_dir[i]))
            continue
        pred_feature = []  # 记录所有预测的人脸特征
        pred_bbox = []  # 记录所有预测的人脸框
        for j in range(len(pred)):  # 一张图片可能不只一个人脸
            pred_feature.append(pred[j].normed_embedding)
            pred_bbox.append(pred[j].bbox)
        pred_feature = np.array(pred_feature, dtype=np.float16 if args.float16 else np.float32)
        result = np.dot(pred_feature, feature)  # 进行匹配
        for j in range(len(result)):  # 一张图片可能不只一个人脸
            feature_argmax = np.argmax(result[j])
            if result[j][feature_argmax] > args.threshold:
                name = column[feature_argmax] + ':{:.2f}'.format(result[j][feature_argmax])
                color = (0, 255, 0)  # 绿色
            else:
                name = 'None_{:.2f}'.format(result[j][feature_argmax])
                color = (0, 0, 255)  # 红色
            # 画人脸框
            image = draw(image, pred_bbox[j], name, color)
        cv2.imshow('predict', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('result_' + image_dir[i], image)


def predict_camera():
    # 加载模型
    model = insightface.app.FaceAnalysis(name='buffalo_l')  # 加载模型，首次运行时会自动下载模型文件到用户下的.insightface文件夹中
    model.prepare(ctx_id=-1 if args.device == 'cpu' else 0, det_size=(args.input_size, args.input_size))  # 模型设置
    # 加载数据库
    df_database = pd.read_csv(args.database_path, dtype=np.float16 if args.float16 else np.float32)
    column = df_database.columns
    feature = df_database.values
    # 打开摄像头
    capture = cv2.VideoCapture(0)
    assert capture.isOpened(), '摄像头打开失败'
    cv2.namedWindow('predict')
    print('|已打开摄像头|')
    # 开始预测
    while capture.isOpened():
        _, image = capture.read()  # 读取摄像头的一帧画面
        pred = model.get(image)
        if pred != []:
            pred_feature = []  # 记录所有预测的人脸特征
            pred_bbox = []  # 记录所有预测的人脸框
            for j in range(len(pred)):  # 一张图片可能不只一个人脸
                pred_feature.append(pred[j].normed_embedding)
                pred_bbox.append(pred[j].bbox)
            pred_feature = np.array(pred_feature, dtype=np.float16 if args.float16 else np.float32)
            result = np.dot(pred_feature, feature)  # 进行匹配
            for j in range(len(result)):  # 一张图片可能不只一个人脸
                feature_argmax = np.argmax(result[j])
                if result[j][feature_argmax] > args.threshold:
                    name = column[feature_argmax] + '_{:.2f}'.format(result[j][feature_argmax])
                    color = (0, 255, 0)  # 绿色
                else:
                    name = 'None_{:.2f}'.format(result[j][feature_argmax])
                    color = (0, 0, 255)  # 红色
                # 画人脸框
                image = draw(image, pred_bbox[j], name, color)
        cv2.imshow('predict', image)
        cv2.waitKey(max(args.camera_time, 1))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_camera() if args.camera else predict()
