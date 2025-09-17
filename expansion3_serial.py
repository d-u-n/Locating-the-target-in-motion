import cv2
import numpy as np
import math
import time
import serial
import threading

# ==== 串口配置 ====
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 115200
SEND_INTERVAL = 0.2  # 5ms控制周期（200Hz）

# ==== 全局变量 ====
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
last_send_time = 0
program_running = False
exit_program = False
received_command = ""

# ==== 状态定义 ====
class State:
    INIT = 0
    FIND_START = 1       # 寻找起始点
    TRACKING = 2         # 跟踪状态

# ==== 配置参数 ====
TARGET_RADIUS_CM = 6.0    # 目标圆半径6cm
A4_WIDTH_CM = 29.7       # A4纸横向宽度29.7cm
A4_HEIGHT_CM = 21.0      # A4纸横向高度21cm
CIRCLE_POINTS = 100      # 圆形轨迹点数

def serial_receive_thread():
    """串口接收线程"""
    global received_command, program_running, exit_program
    while not exit_program:
        try:
            if ser.in_waiting:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw:
                    received_command = raw
                    print(f"收到指令: {received_command}")
                    if received_command == "3":
                        program_running = True
                    elif received_command == "0":
                        program_running = False
        except Exception as e:
            print(f"串口接收错误: {e}")
        time.sleep(0.01)

def send_serial_data(stepx, stepy, center_detected):
    """串口数据发送函数"""
    global last_send_time
    now = time.time()
    if now - last_send_time >= SEND_INTERVAL:
        try:
            if center_detected:
                msg = f"y-step:{stepx:.2f}, {stepy:.2f}, 1.00\r\n"
            else:
                msg = "n-step\r\n"
            
            ser.write(msg.encode('utf-8'))
            print(f"发送: {msg.strip()}")
            last_send_time = now
        except Exception as e:
            print(f"串口发送错误: {e}")

def _order_points(pts):
    """对矩形角点进行排序（左上、右上、右下、左下）"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def detect_perspective_rectangle(frame):
    # 预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edged = cv2.Canny(binary, 50, 150)
    
    # 检测轮廓
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            # 获取矩形四个角点并排序
            src_pts = _order_points(approx.reshape(4, 2))
            
            # 计算横向A4纸尺寸(像素) - 宽度>高度
            target_width_px = int(frame.shape[1] * 0.8)  # 适当缩放
            target_height_px = int(target_width_px * A4_HEIGHT_CM / A4_WIDTH_CM)
            
            # 透视变换目标点（横向A4）
            dst_pts = np.array([
                [0, 0],
                [target_width_px - 1, 0],
                [target_width_px - 1, target_height_px - 1],
                [0, target_height_px - 1]], dtype="float32")
            
            # 计算变换矩阵
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            # 应用透视变换得到横向A4图像
            warped = cv2.warpPerspective(frame, M, (target_width_px, target_height_px))
            
            # 计算中心点(在标准图中的坐标)
            center_px = np.array([[target_width_px // 2, target_height_px // 2]], dtype="float32")
            
            # 转换回原图坐标
            center_original = cv2.perspectiveTransform(
                center_px.reshape(1, 1, 2), M_inv
            )[0][0]
            
            # 计算6cm半径对应的像素距离（基于A4高度比例）
            radius_px = int(TARGET_RADIUS_CM / A4_HEIGHT_CM * target_height_px)
            
            # 生成标准图中的圆形点
            angles = np.linspace(0, 2*np.pi, CIRCLE_POINTS)
            circle_x = center_px[0, 0] + radius_px * np.cos(angles)
            circle_y = center_px[0, 1] + radius_px * np.sin(angles)
            circle_points = np.stack([circle_x, circle_y], axis=1).reshape(1, -1, 2).astype(np.float32)
            
            # 将圆形点转换回原图坐标
            original_circle_points = cv2.perspectiveTransform(circle_points, M_inv)[0]
            
            return {
                'warped': warped,
                'transform_matrix': M,
                'inverse_matrix': M_inv,
                'center_standard': center_px[0],
                'original_circle_points': original_circle_points,
                'contour': approx,
                'center_original': center_original,
                'radius_px': radius_px,
                'standard_size': (target_width_px, target_height_px)
            }
    
    return None

def find_start_point(points, center):
    """找到正上方起始点（x最近，y最小）"""
    start_idx = 0
    min_dist = float('inf')
    for i, pt in enumerate(points):
        if pt[1] < center[1]:  # y坐标在中心上方
            dist = abs(pt[0] - center[0])  # x方向距离
            if dist < min_dist:
                min_dist = dist
                start_idx = i
    return start_idx

def main():
    global exit_program
    
    # 启动串口线程
    threading.Thread(target=serial_receive_thread, daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    current_state = State.INIT
    current_point_idx = 0
    last_control_time = time.time()

    try:
        while not exit_program:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 获取屏幕中心坐标（固定值）
            img_center = (frame.shape[1]//2, frame.shape[0]//2)
            
            result = detect_perspective_rectangle(frame)
            center_detected = result is not None
            stepx, stepy = 0.0, 0.0  # 默认值
            
            # 创建显示图像
            vis = frame.copy()
            
            if result:
                # 获取靶心坐标和轨迹点
                target_center = result['center_original']
                circle_points = result['original_circle_points']
                
                # 绘制所有轨迹点
                for i, pt in enumerate(circle_points):
                    pt_pos = (int(pt[0]), int(pt[1]))
                    if current_state == State.FIND_START and i == current_point_idx:
                        color = (0, 255, 255)  # 黄色
                    elif current_state == State.TRACKING and i == current_point_idx:
                        color = (0, 255, 0)    # 绿色
                    else:
                        color = (200, 200, 200) # 灰色
                    cv2.circle(vis, pt_pos, 4, color, -1)
                
                # 状态处理
                if current_state == State.INIT:
                    current_state = State.FIND_START
                    current_point_idx = find_start_point(circle_points, img_center)
                    print("State: INIT -> FIND_START")
                
                elif current_state == State.FIND_START:
                    target_pt = circle_points[current_point_idx]
                    stepx = target_pt[0] - img_center[0]
                    stepy = target_pt[1] - img_center[1]
                    distance = math.sqrt(stepx**2 + stepy**2)
                    
                    if distance < 10:
                        current_state = State.TRACKING
                        print("State: FIND_START -> TRACKING")
                
                elif current_state == State.TRACKING:
                    if time.time() - last_control_time >= SEND_INTERVAL:
                        next_idx = (current_point_idx + 1) % CIRCLE_POINTS
                        target_pt = circle_points[next_idx]
                        stepx = target_pt[0] - img_center[0]
                        stepy = target_pt[1] - img_center[1]
                        current_point_idx = next_idx
                        last_control_time = time.time()
                
                # 绘制连接线
                current_pt = circle_points[current_point_idx]
                cv2.line(vis, img_center, 
                        (int(current_pt[0]), int(current_pt[1])), 
                        (255, 0, 255), 1)
            
            # 绘制屏幕中心（蓝色）和靶心（红色）
            cv2.circle(vis, img_center, 6, (255, 0, 0), -1)
            if result:
                cv2.circle(vis, (int(target_center[0]), int(target_center[1])), 
                          6, (0, 0, 255), -1)
            
            # 发送串口数据（不论什么状态都发送）
            send_serial_data(stepx, stepy, center_detected)
            
            # 显示状态信息
            state_names = ["INIT", "FIND_START", "TRACKING"]
            cv2.putText(vis, f"State: {state_names[current_state]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"Point: {current_point_idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Target Tracking", vis)
            if cv2.waitKey(1) == ord('q'):
                exit_program = True
    
    except KeyboardInterrupt:
        print("程序终止")
    finally:
        cap.release()
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()