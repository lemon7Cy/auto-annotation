# -*- coding: utf-8 -*-
"""
模式4: 提示词式标注界面
仅背景图，无下方提示区域
从文件名解析类别选项（格式：类别1_类别2_类别3_xxx.jpg）
"""
import os
import cv2
import numpy as np
import json
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QListWidget, QPushButton, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QInputDialog
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


class HintModeWidget(QWidget):
    """提示词式标注组件 - 从文件名解析类别"""
    
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
        self.detected_crops: List[np.ndarray] = []
        
        # 当前文件名解析的类别选项
        self.current_class_options: List[str] = []
        
        # 类别统计
        self.class_counts: Dict[str, int] = {}
        
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
        
        self.model_label = QLabel("未加载模型")
        toolbar.addWidget(self.model_label)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # 说明标签
        hint_label = QLabel("📌 文件名格式：类别1_类别2_类别3_xxx.jpg → 自动解析为可选类别")
        hint_label.setStyleSheet("color: #666; padding: 5px; background: #f5f5f5; border-radius: 3px;")
        layout.addWidget(hint_label)
        
        # 主内容区
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：图像显示区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 大图显示
        self.bg_label = QLabel("加载图片文件夹后显示")
        self.bg_label.setMinimumSize(300, 200)
        self.bg_label.setMaximumHeight(300)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        left_layout.addWidget(self.bg_label, 1)
        
        # 检测到的目标
        crop_group = QGroupBox("检测到的目标（点击选择后从下拉菜单选择类别）")
        crop_layout = QHBoxLayout(crop_group)
        self.crop_scroll = QScrollArea()
        self.crop_scroll.setWidgetResizable(True)
        self.crop_container = QWidget()
        self.crop_grid = QHBoxLayout(self.crop_container)
        self.crop_grid.setAlignment(Qt.AlignLeft)
        self.crop_scroll.setWidget(self.crop_container)
        self.crop_scroll.setMinimumHeight(150)
        self.crop_scroll.setMaximumHeight(250)
        crop_layout.addWidget(self.crop_scroll)
        left_layout.addWidget(crop_group, 1)
        
        # 当前类别选项显示
        options_group = QGroupBox("当前可选类别（从文件名解析）")
        options_layout = QVBoxLayout(options_group)
        self.options_label = QLabel("加载图片后显示")
        self.options_label.setWordWrap(True)
        self.options_label.setStyleSheet("font-size: 14px; padding: 10px;")
        options_layout.addWidget(self.options_label)
        left_layout.addWidget(options_group)
        
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
        
        right_layout.addWidget(img_group)
        
        # 类别统计
        stats_group = QGroupBox("类别统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_list = QListWidget()
        stats_layout.addWidget(self.stats_list)
        right_layout.addWidget(stats_group)
        
        # 按钮区
        btn_layout = QHBoxLayout()
        
        self.complete_btn = QPushButton("✓ 完成当前图片")
        self.complete_btn.setStyleSheet("background: #4CAF50; color: white; font-weight: bold;")
        self.complete_btn.clicked.connect(self._manual_complete)
        btn_layout.addWidget(self.complete_btn)
        
        self.skip_btn = QPushButton("跳过")
        self.skip_btn.clicked.connect(self._next_image)
        btn_layout.addWidget(self.skip_btn)
        
        right_layout.addLayout(btn_layout)
        
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
        
        # 扫描图片文件
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
        
        # 跳到第一个未完成的
        for i in range(len(self.image_files)):
            if i not in self.completed_images:
                self.image_list.setCurrentRow(i)
                break
        else:
            if self.image_files:
                self.image_list.setCurrentRow(0)
    
    def _save_progress(self):
        """保存标注进度"""
        if not self.data_folder:
            return
        
        progress_file = os.path.join(self.data_folder, ".hint_progress.json")
        data = {
            'completed_images': list(self.completed_images),
            'class_counts': self.class_counts
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
        
        progress_file = os.path.join(self.data_folder, ".hint_progress.json")
        if not os.path.exists(progress_file):
            return
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.completed_images = set(data.get('completed_images', []))
            self.class_counts = data.get('class_counts', {})
            
            # 更新列表显示
            for i in range(self.image_list.count()):
                if i in self.completed_images:
                    item = self.image_list.item(i)
                    if item and not item.text().startswith('✓'):
                        item.setText(f"✓ {os.path.basename(self.image_files[i])}")
            
            self._update_stats()
        except Exception as e:
            print(f"加载进度失败: {e}")
    
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
        
        # 从文件名解析类别选项（格式：类型1_类型2_类型3_xxx.jpg）
        filename = os.path.basename(img_path)
        name_part = os.path.splitext(filename)[0]
        parts = name_part.split('_')
        
        # 过滤：长度<=8的保留，或包含中文的保留
        self.current_class_options = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 保留短字符串或包含非ASCII字符（如中文）
            if len(part) <= 8 or not part.isascii():
                self.current_class_options.append(part)
        
        # 更新类别选项显示
        if self.current_class_options:
            self.options_label.setText("  |  ".join(self.current_class_options))
        else:
            self.options_label.setText("（未从文件名解析到类别选项）")
        
        # 显示图片
        self._display_bg(self.current_image)
        
        # 运行检测
        self.detected_crops = []
        
        if self.detector:
            try:
                detections = self.detector.detect(img_path)
                
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = self.current_image[y1:y2, x1:x2].copy()
                    
                    if crop.size > 0:
                        self.detected_crops.append(crop)
                
            except Exception as e:
                print(f"检测错误: {e}")
        
        # 显示裁剪结果
        self._display_crops()
    
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
    
    def _display_crops(self):
        """显示检测到的目标"""
        while self.crop_grid.count():
            item = self.crop_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, crop in enumerate(self.detected_crops):
            label = ClickableImageLabel(i)
            label.clicked.connect(self._on_crop_clicked)
            
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(pixmap)
            self.crop_grid.addWidget(label)
    
    def _on_crop_clicked(self, idx: int):
        """点击目标 - 弹出类别选择"""
        if idx < 0 or idx >= len(self.detected_crops):
            return
        
        crop_img = self.detected_crops[idx]
        
        # 如果有类别选项，使用下拉选择
        if self.current_class_options:
            class_name, ok = QInputDialog.getItem(
                self, "选择类别",
                "请选择该目标的类别:",
                self.current_class_options,
                0,
                editable=True  # 允许手动输入
            )
        else:
            class_name, ok = QInputDialog.getText(
                self, "输入类别名",
                "请输入该目标的类别名:",
                text="class_1"
            )
        
        if not ok or not class_name.strip():
            return
        
        class_name = class_name.strip()
        
        # 保存
        self._save_crop(crop_img, class_name)
        
        # 移除已分类的
        del self.detected_crops[idx]
        self._display_crops()
        
        self._update_stats()
        
        # 如果都分类完了，下一张
        if not self.detected_crops:
            self.completed_images.add(self.current_idx)
            item = self.image_list.item(self.current_idx)
            if item:
                item.setText(f"✓ {os.path.basename(self.image_files[self.current_idx])}")
            self._save_progress()
            self._next_image()
    
    def _save_crop(self, crop_img: np.ndarray, class_name: str):
        """保存到分类文件夹"""
        if not self.output_folder:
            self.output_folder = os.path.join(self.data_folder, "datasets")
        
        class_folder = os.path.join(self.output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        existing = len([f for f in os.listdir(class_folder) if f.startswith('crop_')])
        crop_path = os.path.join(class_folder, f"crop_{existing + 1:04d}.jpg")
        cv_imwrite(crop_path, crop_img)
        
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
    
    def _update_stats(self):
        """更新统计"""
        self.stats_list.clear()
        for name, count in sorted(self.class_counts.items()):
            self.stats_list.addItem(f"{name}: {count} 张")
    
    def _manual_complete(self):
        """手动完成当前图片，跳过剩余目标"""
        if self.current_idx < 0:
            return
        
        # 标记为已完成
        self.completed_images.add(self.current_idx)
        item = self.image_list.item(self.current_idx)
        if item:
            item.setText(f"✓ {os.path.basename(self.image_files[self.current_idx])}")
        
        # 保存进度
        self._save_progress()
        
        # 下一张
        self._next_image()
    
    def _prev_image(self):
        if self.current_idx > 0:
            self.image_list.setCurrentRow(self.current_idx - 1)
    
    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.image_list.setCurrentRow(self.current_idx + 1)

