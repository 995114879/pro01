import cv2
import torch
from a3_7.yolov5.models.common import DetectMultiBackend
from deep_sort_realtime.deepsort_tracker import DeepSort

# 加载 YOLOv5 模型
# torch.hub --> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测 --> 只要求你加载的模型文件优hubconf.py文件
model = torch.hub.load(
    repo_or_dir="../yolov5",  # 给定github上的项目名称或本地文件夹路径
    model="yolov5s",  # 给定模型文件,其实就是hubconf.py中的方法名
    source='local'  # 加载的代码/模型来源：可选：github、local
)
use_gpu = True
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=30)  # 参数可调

# 打开本地摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面")
        break

    # YOLOv5 目标检测
    results = model(frame, augment=False)  # 检测图像
    detections = []

    # 提取每个检测结果的框、置信度和类别
    for *box, conf, cls in results.xyxy[0]:  # 遍历检测结果
        if int(cls) == 0 and conf > 0.4:  # cls == 0 表示“person”类
            x1, y1, x2, y2 = map(int, box)  # 坐标转换为整数
            detections.append(([x1, y1, x2, y2], conf.item(), 'person'))  # 添加检测框

    # 目标跟踪
    tracks = tracker.update_tracks(detections, frame=frame)

    # 显示跟踪结果
    for track in tracks:
        if not track.is_confirmed():  # 忽略未确认的轨迹
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltwh()  # 左上角和右下角坐标
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Camera Tracking", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
