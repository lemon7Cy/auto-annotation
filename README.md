# Auto Annotation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

YOLOv5 半自动标注工具 + 孪生分类数据集制作工具

## ✨ 功能特性

### 🎯 YOLO 标注模式
- 支持 ONNX 模型推理预标注
- 可视化边界框编辑
- 点击放置固定尺寸框（支持类别尺寸记忆）
- 自动切换下一张，提高标注效率
- 支持中文路径

### 🔗 孪生分类模式
用于制作孪生网络（Siamese Network）分类训练数据集，支持四种工作模式：

| 模式 | 名称 | 说明 |
|:---:|:---:|:---|
| 1/3 | 分离式 | 背景图 + 多个提示图（icon）分开存放 |
| 2 | 单图式 | 一张图包含上方目标 + 下方提示 |
| 4 | 提示词式 | 仅背景图，从文件名解析类别 |

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/auto-annotation.git
cd auto-annotation

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表
```
PyQt5>=5.15
opencv-python>=4.5
numpy>=1.20
onnxruntime>=1.10
ddddocr>=1.4  # 可选，用于OCR模式
```

## 🚀 快速开始

```bash
python main.py
```

## 📖 使用指南

### YOLO 标注模式

1. **新建项目** → 选择工作目录
2. **添加图片文件夹** → 导入待标注图片
3. **加载模型**（可选）→ 使用 ONNX 模型预标注
4. **标注** → 拖拽框选或点击放置
5. **保存** → 自动生成 YOLO 格式标签

### 孪生分类 - 模式 1/3：分离式

**输入格式：**
```
数据文件夹/
├── sample1/
│   ├── bg.jpg        # 背景图
│   ├── 苹果.png      # 提示图（文件名=类别名）
│   └── 香蕉.png
└── sample2/
    ├── bg.jpg
    └── 猫.png
```

**操作流程：**
1. 加载 YOLO 模型
2. 加载数据文件夹
3. 点击检测目标 → 点击提示图匹配 → 确认类别名
4. 自动保存并跳转下一个

**输出格式：**
```
数据文件夹/
├── datasets/
│   ├── 苹果/
│   │   ├── crop_0001.jpg
│   │   └── hint.png
│   └── 香蕉/
│       └── crop_0001.jpg
└── .siamese_progress.json
```

### 孪生分类 - 模式 2：单图式

**输入格式：**
```
图片文件夹/
├── image001.jpg    # 上方：待分类目标，下方：提示图
└── image002.jpg
```

**操作流程：**
1. 加载 YOLO 模型
2. 加载图片文件夹
3. 调整分割比例（默认 0.7）
4. 点击上方目标 → 点击下方提示图 → 输入类别名

### 孪生分类 - 模式 4：提示词式

**输入格式：**
```
图片文件夹/
├── 苹果_香蕉_橙子_abc123.jpg
└── 猫_狗_兔子_def456.jpg
```

文件名中 `_` 分隔的短词（≤8字符或中文）自动解析为类别选项。

**操作流程：**
1. 加载 YOLO 模型
2. 加载图片文件夹
3. 点击检测目标 → 从下拉菜单选择类别
4. 点击「完成当前图片」跳过剩余目标

## 🔧 配置

### 检测阈值
在 `yolo_detector.py` 中修改：
```python
self.conf_threshold = 0.25
self.iou_threshold = 0.45
```

## 📁 项目结构

```
auto-annotation/
├── main.py                    # 入口
├── main_window.py             # 主窗口
├── image_canvas.py            # 标注画布
├── yolo_detector.py           # YOLO 推理
├── siamese_widget.py          # 孪生分类容器
├── siamese_folder_widget.py   # 模式 1/3
├── single_image_widget.py     # 模式 2
├── hint_mode_widget.py        # 模式 4
└── project_manager.py         # 项目管理
```

## 📄 License

MIT License

## 🙏 鸣谢

- [ddddocr](https://github.com/sml2h3/ddddocr) - 感谢 [@sml2h3](https://github.com/sml2h3) 提供的 OCR 识别库
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5) - YOLO 目标检测框架

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

