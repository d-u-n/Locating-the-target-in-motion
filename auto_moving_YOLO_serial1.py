import cv2
import numpy as np
import onnxruntime as ort
import time
import serial
import threading

# ==== 串口配置 ====
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 115200
SEND_INTERVAL = 0.1  # 每隔0.1秒发送一次

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
last_send_time = 0

# ==== 全局控制标志 ====
program_running = False  # 程序运行状态标志
exit_program = False     # 退出程序标志

# ==== 串口发送结构 ====
latest_data = {
    "step": [0],
    "center_detected": False
}

# ==== 串口接收状态 ====
received_command = ""

# ==== 串口接收线程 ====
def serial_receive_thread():
    global received_command, program_running, exit_program
    while not exit_program:
        try:
            if ser.in_waiting:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw:
                    received_command = raw
                    print(f"收到串口指令: {received_command}")
                    
                    # 处理控制命令
                    if received_command == "2":
                        program_running = True
                        print("程序开始运行")
                    elif received_command == "0":
                        program_running = False
                        print("程序停止运行")
                    else:
                        handle_serial_command(received_command)
        except Exception as e:
            print(f"串口接收错误: {e}")
        time.sleep(0.01)

# ==== 接收指令解析函数 ====
def handle_serial_command(cmd):
    if cmd.startswith("CMD:"):
        action = cmd[4:]
        print(f"执行命令动作: {action}")
    elif cmd.startswith("LED:"):
        val = cmd[4:]
        print(f"控制LED状态: {val}")
    elif cmd.startswith("PWM:"):
        pwm_val = cmd[4:]
        print(f"设置PWM占空比: {pwm_val}")
    else:
        print(f"未识别指令: {cmd}")

# ==== 串口发送函数 ====
def send_serial_data():
    global last_send_time
    now = time.time()
    if now - last_send_time >= SEND_INTERVAL:
        try:
            if latest_data["center_detected"]:
                # 当检测到中心点时，发送step
                step = latest_data["step"][0]
                msg = f"y-step:{step:.2f}, 0.00, 1.00\r\n"
            else:
                # 未检测到中心点时，只发送n-step
                msg = "n-step\r\n"
            
            ser.write(msg.encode('utf-8'))
            print(f"发送: {msg.strip()}")
            last_send_time = now
        except Exception as e:
            print(f"串口发送错误: {e}")

# 状态定义
class State:
    INIT = 0
    YOLO_DETECTION = 1
    PRECISE_ALIGNMENT = 2
    COMPLETED = 3

def calculate_distance_cm(pixel_width):
    focal_pixel = 565.2
    real_width_cm = 17.4
    if pixel_width == 0:
        return 0
    return (focal_pixel * real_width_cm) / pixel_width

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect

def _compute_perspective_center(pts):
    center_x = (pts[0][0] + pts[2][0]) / 2
    center_y = (pts[0][1] + pts[2][1]) / 2
    return (center_x, center_y)

def detect_perspective_rectangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edged = cv2.Canny(binary, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rect = cv2.minAreaRect(approx)
        (center_x, center_y), (w, h), angle = rect
        
        if len(approx) == 4:
            src_pts = _order_points(approx.reshape(4, 2))
            center = _compute_perspective_center(src_pts)
            return center, w
    
    return None, 0

class TargetDetector:
    def __init__(self):
        # YOLO模型初始化
        model_path = "/home/dun/E/detect/black/YOLOv5-Lite-master/best.onnx"  # 替换为你的模型路径
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(model_path, so)
        
        # 模型参数
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        anchors = [[10, 13, 16, 30, 33, 23], 
                  [30, 61, 62, 45, 59, 119], 
                  [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        
        # 状态机初始化
        self.state = State.INIT
        self.last_state_change = time.time()
        
    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    
    def cal_outputs(self, outs):
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_w / self.stride[i]), int(self.model_h / self.stride[i])
            length = int(self.na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)
    
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs
    
    def post_process(self, outputs, img_h, img_w):
        conf = outputs[:,4].tolist()
        c_x = outputs[:,0]/self.model_w*img_w
        c_y = outputs[:,1]/self.model_h*img_h
        w = outputs[:,2]/self.model_w*img_w
        h = outputs[:,3]/self.model_h*img_h
        cls_id = np.argmax(outputs[:,5:], axis=1)
    
        p_x1 = np.expand_dims(c_x-w/2,-1)
        p_y1 = np.expand_dims(c_y-h/2,-1)
        p_x2 = np.expand_dims(c_x+w/2,-1)
        p_y2 = np.expand_dims(c_y+h/2,-1)
        areas = np.concatenate((p_x1,p_y1,p_x2,p_y2), axis=-1)
        
        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, 0.5, 0.4)
        if len(ids)>0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        return [], [], []
    
    def yolo_detect(self, img0):
        # 图像预处理
        img = cv2.resize(img0, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    
        # 模型推理
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        outs = self.cal_outputs(outs)
    
        # 检测框计算
        img_h, img_w = img0.shape[:2]
        boxes, confs, ids = self.post_process(outs, img_h, img_w)
        
        if len(boxes) > 0:
            # 只取第一个检测结果
            x1, y1, x2, y2 = boxes[0].astype(np.int32)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return (center_x, center_y)
        return None
    
    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        image_center = (img_w // 2, img_h // 2)
        
        # 重置串口数据
        latest_data["center_detected"] = False
        latest_data["step"] = [0]
        
        if self.state == State.INIT:
            self.state = State.YOLO_DETECTION
            self.last_state_change = time.time()
        
        elif self.state == State.YOLO_DETECTION:
            target_center = self.yolo_detect(frame)
            
            if target_center:
                # 计算水平像素距离
                pixel_dist = abs(target_center[0] - image_center[0])
                
                # 计算step值
                step = (135/640)*(image_center[0]-target_center[0])
                latest_data["step"] = [step]
                latest_data["center_detected"] = True
                
                if pixel_dist < 50:
                    self.state = State.PRECISE_ALIGNMENT
                    self.last_state_change = time.time()
        
        elif self.state == State.PRECISE_ALIGNMENT:
            target_center, w = detect_perspective_rectangle(frame)
            
            if target_center:
                pixel_dist = abs(target_center[0] - image_center[0])
                distance_cm = calculate_distance_cm(w)
                
                # 计算step
                step = (135/640)*(image_center[0]-target_center[0])
                
                latest_data["step"] = [step]
                latest_data["center_detected"] = True
                
                if pixel_dist < 10:
                    self.state = State.COMPLETED
                    self.last_state_change = time.time()
        
        elif self.state == State.COMPLETED:
            target_center, w = detect_perspective_rectangle(frame)
            
            if target_center:
                pixel_dist = abs(target_center[0] - image_center[0])
                distance_cm = calculate_distance_cm(w)
                
                # 计算step
                step = (135/640)*(image_center[0]-target_center[0])
                
                latest_data["step"] = [step]
                latest_data["center_detected"] = True
        
        return self.state

def main():
    # 启动串口接收线程
    threading.Thread(target=serial_receive_thread, daemon=True).start()

    detector = TargetDetector()
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_state = detector.process_frame(frame)
            
            # 发送串口数据
            send_serial_data()
            
            if current_state == State.COMPLETED:
                print("Alignment completed successfully!")
            
            if cv2.waitKey(1) == ord('q'):
                exit_program = True
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("中断退出")
    finally:
        cap.release()
        ser.close()
        print("程序已退出")

if __name__ == "__main__":
    main()