# 全国大学生电子设计竞赛 (2025) E题 - 视觉追踪系统

[![Python](https://img.shields.io/badge/Python-3.11.2-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch->=1.8.0-orange)](https://pytorch.org/)
[![YOLOv5-Lite](https://img.shields.io/badge/YOLO-v5--Lite-red)](https://github.com/ppogg/YOLOv5-Lite)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-ff69b4)](https://www.raspberrypi.com/)

本项目为2025年全国大学生电子设计竞赛E题的视觉解决方案。系统通过摄像头实时检测运动中的黑色矩形框与固定半径（6cm）的圆形目标，计算其中心坐标，并通过串口通信将坐标数据发送至二维云台控制系统，以实现自动追踪与瞄准。

## 当前功能概述

-   **多目标检测**: 同时处理黑色矩形框与圆形目标。
-   **混合算法策略**: 结合了传统OpenCV图像处理与轻量化YOLOv5-Lite深度学习模型，以应对高速运动导致的“掉框”问题。
-   **实时性能**: 在目标设备上运行可达 **30+ FPS**，满足实时性要求。
-   **坐标通信**: 将计算出的目标坐标通过串口发送给下位机（云台控制器）。
-   **系统控制**: 通过串口指令实现不同功能模式的动态切换与生命周期管理。

## 技术栈

-   **编程语言**: Python 3.11.2
-   **核心视觉库**: OpenCV 4.10.0
-   **深度学习框架**: PyTorch (>=1.8.0)
-   **硬件通信**: `pyserial`
-   **部署平台**: Raspberry Pi

## 解决方案架构

针对高速运动目标导致的“掉框”这一核心难点，本项目创新性地采用了 **“粗定位+精定位”** 的混合策略与**状态机**控制逻辑，兼具速度与精度。

1.  **粗定位 (YOLOv5-Lite)**: 轻量化神经网络快速处理整帧图像，鲁棒地检测出运动黑框的大致区域（Bounding Box），有效解决因运动模糊、边缘断裂导致的传统方法失效问题。
2.  **精定位 (OpenCV)**: 在YOLO提供的候选区域内，使用传统的图像处理技术（如轮廓查找、霍夫圆变换、矩计算等）进行像素级精确分析，最终定位黑框和圆形的中心坐标，精度远高于单独使用YOLO。
3.  **状态机控制**: 通过状态机逻辑智能地在“纯OpenCV模式”和“YOLO辅助模式”之间切换或融合，确保系统在目标低速、高速等各种场景下均能保持稳定跟踪。

## 性能与效果

-   **运行平台**: Raspberry Pi 5
-   **处理帧率**: **> 30 FPS** 稳定运行，满足实时性要求。
-   **准确性**: 在竞赛测试场景下，系统能持续稳定跟踪目标，无频繁跟丢现象。*（注：因竞赛开发周期限制，未进行正式的mAP/Precision/Recall量化测试，但实际效果显著优于单一方法）*
-   **延迟**: *（注：端到端系统延迟是云台控制的关键指标，建议后续使用高帧率相机进行定量测量以进一步优化）*

## 效果展示

我提供了完整的系统运行演示视频，直观展示了视觉检测的效果：

[点击此处观看完整演示视频 (Bilibili)](https://www.bilibili.com/video/BV1BEpxz1EXc/?vd_source=edfde277e09b835c3d6a052e455a11e6)**

*视频内容包含：*
- *混合检测策略（YOLO+OpenCV）的实际效果*
- *高速运动目标下的稳定跟踪表现*

## 安装

1.  **克隆本仓库**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **创建Python虚拟环境（推荐）**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

### 依赖项 (requirements.txt)
项目依赖的Python库详细列表如下，已固定关键版本以确保环境一致性：

```txt
# base ----------------------------------------
matplotlib>=3.2.2
numpy==1.18.0
opencv-python==4.10.0.84
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.41.0
pyserial  # 用于串口通信

# logging -------------------------------------
tensorboard>=2.4.1
# wandb

# plotting ------------------------------------
seaborn>=0.11.0
pandas

# export --------------------------------------
coremltools>=4.1
onnx>=1.9.1
scikit-learn  # for coreml quantization

# extras --------------------------------------
thop  # FLOPS computation
pycocotools>=2.0  # COCO mAP
```

## 使用教程

本项目设计为一个**常驻的守护进程**，通过串口接收指令来动态控制不同的视觉任务。系统启动后即处于监听状态。

### 串口指令控制

使用任意串口工具（如`screen`, `minicom`, 或Python的`pyserial`库）向树莓派发送单个数字字符即可切换模式：

| 指令 | 运行程序 | 功能描述 |
| :--- | :--- | :--- |
| `1` | `basic_YOLO_serial.py` | 启动**基础功能**：检测运动黑框并定位中心 |
| `2` | `auto_moving_YOLO_serial1.py` | 启动**扩展功能1&2**：自动移动跟踪等高级功能 |
| `3` | `expansion3_serial.py` | 启动**扩展功能3**：特定任务模式 |
| `0` | (无) | **停止**当前正在运行的所有任务，系统返回待命状态 |

### 快速开始

1.  确保树莓派已正确连接摄像头和串口线。
2.  系统已配置为**开机自启动**（启动后自动运行监听程序）。
3.  通过串口发送指令 `1`、`2` 或 `3` 即可启动相应的视觉处理程序。
4.  发送指令 `0` 可随时停止当前任务。

## 项目结构

```
project-root/
├── basic_YOLO_serial.py      # 基础功能：目标检测与中心定位
├── auto_moving_YOLO_serial1.py # 扩展功能1&2：自动跟踪
├── expansion3_serial.py      # 扩展功能3：特定任务模式
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明文档
└── ...                       # 模型权重、配置文件等
```
