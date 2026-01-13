# -*- coding: utf-8 -*-
"""
单图匹配标注界面
一张图片包含上方目标区域 + 下方提示图区域
"""
import os
import cv2
import numpy as np
import json
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QListWidget, QPushButton, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QDoubleSpinBox, QFormLayout,
    QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal


def cv_imread(filepath: str):
    """读取图片，支持中文路径"""
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(filepath: str, img: np.ndarray):
    """保存图片，支持中文路径"""
    ext = os.path.splitext(filepath)[1]
    cv2.imencode(ext, img)[1].tofile(filepath)


def get_image_hash(img: np.ndarray) -> str:
    """计算图片hash作为类别标识"""
    # 缩小图片并计算简单hash
    small = cv2.resize(img, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    bits = (gray > avg).flatten()
    # 转为16进制字符串
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return f"cls_{hash_val:016x}"[:12]  # 取前12位


class ClickableImageLabel(QLabel):
    """可点击的图片标签"""
    clicked = pyqtSignal(int)
    
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFrameStyle(QFrame.Box)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.selected = False
        self._update_style()
    
    def set_selected(self, selected: bool):
        self.selected = selected
        self._update_style()
    
    def _update_style(self):
        if self.selected:
            self.setStyleSheet("border: 3px solid #00aa00; background: #e0ffe0;")
        else:
            self.setStyleSheet("border: 1px solid #999; background: #fff;")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)


class SingleImageWidget(QWidget):
    """单图匹配标注组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.data_folder: str = ""
        self.output_folder: str = ""
        self.image_files: List[str] = []
        self.current_idx: int = -1
        
        # YOLO 检测器
        self.detector = None
        
        # 当前图片数据
        self.current_image: Optional[np.ndarray] = None
        self.upper_crops: List[np.ndarray] = []  # 上方目标
        self.lower_crops: List[np.ndarray] = []  # 下方提示图
        
        # 分割比例（默认0.7，即70%以上是目标区，30%以下是提示区）
        self.split_ratio = 0.7
        
        # 选择状态
        self.selected_upper_idx: int = -1
        
        # 类别统计
        self.class_counts: Dict[str, int] = {}
        
        # 类别名称记忆 {位置索引: 类别名}
        self.class_name_memory: Dict[int, str] = {}
        
        # 已完成的图片索引
        self.completed_images: set = set()
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 顶部工具栏
        toolbar = QHBoxLayout()
        
        self.load_model_btn = QPushButton("加载YOLO模型")
        self.load_model_btn.clicked.connect(self._load_model)
        toolbar.addWidget(self.load_model_btn)
        
        self.load_data_btn = QPushButton("加载图片文件夹")
        self.load_data_btn.clicked.connect(self._load_data_folder)
        toolbar.addWidget(self.load_data_btn)
        
        # 分割比例设置
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.5, 0.9)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(0.7)
        self.split_spin.valueChanged.connect(self._on_split_changed)
        toolbar.addWidget(QLabel("分割比例:"))
        toolbar.addWidget(self.split_spin)
        
        self.model_label = QLabel("未加载模型")
        toolbar.addWidget(self.model_label)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # 主内容区
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：图像显示区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 大图显示
        self.bg_label = QLabel("加载图片文件夹后显示")
        self.bg_label.setMinimumSize(300, 150)
        self.bg_label.setMaximumHeight(250)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        left_layout.addWidget(self.bg_label, 1)
        
        # 上方目标区域（待分类）
        upper_group = QGroupBox("上方目标（点击选择）")
        upper_layout = QHBoxLayout(upper_group)
        self.upper_scroll = QScrollArea()
        self.upper_scroll.setWidgetResizable(True)
        self.upper_container = QWidget()
        self.upper_grid = QHBoxLayout(self.upper_container)
        self.upper_grid.setAlignment(Qt.AlignLeft)
        self.upper_scroll.setWidget(self.upper_container)
        self.upper_scroll.setMinimumHeight(120)
        self.upper_scroll.setMaximumHeight(200)
        upper_layout.addWidget(self.upper_scroll)
        left_layout.addWidget(upper_group, 1)
        
        # 下方提示区域（作为类别）
        lower_group = QGroupBox("下方提示图（点击匹配，按顺序编号为类别）")
        lower_layout = QHBoxLayout(lower_group)
        self.lower_scroll = QScrollArea()
        self.lower_scroll.setWidgetResizable(True)
        self.lower_container = QWidget()
        self.lower_grid = QHBoxLayout(self.lower_container)
        self.lower_grid.setAlignment(Qt.AlignLeft)
        self.lower_scroll.setWidget(self.lower_container)
        self.lower_scroll.setMinimumHeight(120)
        self.lower_scroll.setMaximumHeight(200)
        lower_layout.addWidget(self.lower_scroll)
        left_layout.addWidget(lower_group, 1)
        
        splitter.addWidget(left_widget)
        
        # 右侧：列表和统计
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 图片列表
        img_group = QGroupBox("图片列表")
        img_layout = QVBoxLayout(img_group)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        img_layout.addWidget(self.image_list)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self._next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        img_layout.addLayout(nav_layout)
        
        # 标注进度
        self.progress_label = QLabel("进度: 0/0 (0%)")
        self.progress_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; padding: 5px;")
        img_layout.addWidget(self.progress_label)
        
        right_layout.addWidget(img_group)
        
        # 类别统计
        stats_group = QGroupBox("类别统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_list = QListWidget()
        stats_layout.addWidget(self.stats_list)
        right_layout.addWidget(stats_group)
        
        self.skip_btn = QPushButton("跳过当前图片")
        self.skip_btn.clicked.connect(self._next_image)
        right_layout.addWidget(self.skip_btn)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 250])
    
    def _load_model(self):
        """加载YOLO ONNX模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO ONNX模型", "", "ONNX模型 (*.onnx)"
        )
        if not model_path:
            return
        
        try:
            from yolo_detector import YOLODetector
            self.detector = YOLODetector(model_path)
            self.model_label.setText(f"模型: {os.path.basename(model_path)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
    
    def _load_data_folder(self):
        """加载图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        
        self.data_folder = folder
        self.image_files = []
        
        # 扫描 jpg 文件
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.image_files.append(os.path.join(folder, f))
        
        # 更新列表
        self.image_list.clear()
        for i, path in enumerate(self.image_files):
            name = os.path.basename(path)
            if i in self.completed_images:
                name = f"✓ {name}"
            self.image_list.addItem(name)
        
        # 输出目录 = 数据文件夹/datasets
        self.output_folder = os.path.join(folder, "datasets")
        
        # 加载进度
        self._load_progress()
        
        # 更新进度显示
        self._update_progress()
        
        # 跳到第一个未完成的
        for i in range(len(self.image_files)):
            if i not in self.completed_images:
                self.image_list.setCurrentRow(i)
                break
        else:
            if self.image_files:
                self.image_list.setCurrentRow(0)
    
    def _save_progress(self):
        """保存标注进度（自动保存，不弹窗）"""
        if not self.data_folder:
            return
        
        progress_file = os.path.join(self.data_folder, ".single_progress.json")
        data = {
            'completed_images': list(self.completed_images),
            'class_counts': self.class_counts,
            'class_name_memory': {str(k): v for k, v in self.class_name_memory.items()}
        }
        
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存进度失败: {e}")
    
    def _load_progress(self):
        """加载标注进度"""
        if not self.data_folder:
            return
        
        progress_file = os.path.join(self.data_folder, ".single_progress.json")
        if not os.path.exists(progress_file):
            return
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.completed_images = set(data.get('completed_images', []))
            self.class_counts = data.get('class_counts', {})
            self.class_name_memory = {int(k): v for k, v in data.get('class_name_memory', {}).items()}
            
            # 更新列表显示
            for i in range(self.image_list.count()):
                if i in self.completed_images:
                    item = self.image_list.item(i)
                    if item and not item.text().startswith('✓'):
                        item.setText(f"✓ {os.path.basename(self.image_files[i])}")
            
            self._update_stats()
        except Exception as e:
            print(f"加载进度失败: {e}")
    
    def _on_split_changed(self, value):
        """分割比例变化"""
        self.split_ratio = value
        if self.current_idx >= 0:
            self._process_current_image()
    
    def _on_image_selected(self, row: int):
        """选择图片"""
        if row < 0 or row >= len(self.image_files):
            return
        
        self.current_idx = row
        self._process_current_image()
    
    def _process_current_image(self):
        """处理当前图片"""
        if self.current_idx < 0:
            return
        
        img_path = self.image_files[self.current_idx]
        self.current_image = cv_imread(img_path)
        
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        split_y = int(h * self.split_ratio)
        
        # 显示图片（带分割线）
        display_img = self.current_image.copy()
        cv2.line(display_img, (0, split_y), (w, split_y), (0, 255, 0), 2)
        self._display_bg(display_img)
        
        # 运行检测
        self.upper_crops = []
        self.lower_crops = []
        
        if self.detector:
            try:
                detections = self.detector.detect(img_path)
                
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    center_y = (y1 + y2) / 2
                    crop = self.current_image[y1:y2, x1:x2].copy()
                    
                    if crop.size > 0:
                        if center_y < split_y:
                            self.upper_crops.append(crop)
                        else:
                            self.lower_crops.append(crop)
                
            except Exception as e:
                print(f"检测错误: {e}")
        
        # 显示裁剪结果
        self._display_upper()
        self._display_lower()
        
        # 重置选择
        self.selected_upper_idx = -1
    
    def _display_bg(self, img: np.ndarray):
        """显示背景图"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled = pixmap.scaled(
            self.bg_label.width() - 10,
            self.bg_label.height() - 10,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.bg_label.setPixmap(scaled)
    
    def _display_upper(self):
        """显示上方目标"""
        while self.upper_grid.count():
            item = self.upper_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, crop in enumerate(self.upper_crops):
            label = ClickableImageLabel(i)
            label.clicked.connect(self._on_upper_clicked)
            
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(pixmap)
            self.upper_grid.addWidget(label)
    
    def _display_lower(self):
        """显示下方提示图"""
        while self.lower_grid.count():
            item = self.lower_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, crop in enumerate(self.lower_crops):
            label = ClickableImageLabel(i)
            label.clicked.connect(self._on_lower_clicked)
            
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(pixmap)
            label.setToolTip(f"类别 {i + 1}")
            self.lower_grid.addWidget(label)
    
    def _on_upper_clicked(self, idx: int):
        """点击上方目标"""
        self.selected_upper_idx = idx
        
        for i in range(self.upper_grid.count()):
            widget = self.upper_grid.itemAt(i).widget()
            if isinstance(widget, ClickableImageLabel):
                widget.set_selected(i == idx)
    
    def _on_lower_clicked(self, idx: int):
        """点击下方提示图 - 保存匹配"""
        if self.selected_upper_idx < 0 or self.selected_upper_idx >= len(self.upper_crops):
            QMessageBox.warning(self, "提示", "请先选择上方的目标图")
            return
        
        if idx < 0 or idx >= len(self.lower_crops):
            return
        
        upper_img = self.upper_crops[self.selected_upper_idx]
        lower_img = self.lower_crops[idx]
        
        # 检查是否有记忆的类别名
        default_name = self.class_name_memory.get(idx, f"class_{idx + 1}")
        
        # 弹出输入框让用户输入/确认类别名
        class_name, ok = QInputDialog.getText(
            self, "输入类别名", 
            f"请输入第{idx + 1}个提示图对应的类别名:",
            text=default_name
        )
        
        if not ok or not class_name.strip():
            return
        
        class_name = class_name.strip()
        
        # 记忆这个位置的类别名，下次自动填充
        self.class_name_memory[idx] = class_name
        
        # 保存
        self._save_to_class(upper_img, lower_img, class_name)
        
        # 移除已分类的
        del self.upper_crops[self.selected_upper_idx]
        self.selected_upper_idx = -1
        self._display_upper()
        
        self._update_stats()
        
        # 如果都分类完了，下一张
        if not self.upper_crops:
            # 标记为已完成
            self.completed_images.add(self.current_idx)
            item = self.image_list.item(self.current_idx)
            if item:
                item.setText(f"✓ {os.path.basename(self.image_files[self.current_idx])}")
            # 自动保存进度
            self._save_progress()
            self._update_progress()
            self._next_image()
    
    def _save_to_class(self, upper_img: np.ndarray, lower_img: np.ndarray, class_name: str):
        """保存到分类文件夹"""
        # 使用 datasets 子目录
        if not self.output_folder:
            self.output_folder = os.path.join(self.data_folder, "datasets")
        
        class_folder = os.path.join(self.output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # 保存上方目标图
        existing = len([f for f in os.listdir(class_folder) if f.startswith('crop_')])
        crop_path = os.path.join(class_folder, f"crop_{existing + 1:04d}.jpg")
        cv_imwrite(crop_path, upper_img)
        
        # 保存提示图（覆盖）
        hint_path = os.path.join(class_folder, "hint.jpg")
        if not os.path.exists(hint_path):
            cv_imwrite(hint_path, lower_img)
        
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
    
    def _update_stats(self):
        """更新统计"""
        self.stats_list.clear()
        for name, count in sorted(self.class_counts.items()):
            self.stats_list.addItem(f"{name}: {count} 张")
    
    def _update_progress(self):
        """更新标注进度"""
        total = len(self.image_files)
        completed = len(self.completed_images)
        if total > 0:
            percent = completed * 100 // total
            self.progress_label.setText(f"进度: {completed}/{total} ({percent}%)")
        else:
            self.progress_label.setText("进度: 0/0 (0%)")
    
    def _prev_image(self):
        if self.current_idx > 0:
            self.image_list.setCurrentRow(self.current_idx - 1)
    
    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.image_list.setCurrentRow(self.current_idx + 1)
