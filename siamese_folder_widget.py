# -*- coding: utf-8 -*-
"""
分离式孪生分类标注界面
输入: 文件夹（bg.jpg + icon*.png）
适用于模式1（图标匹配）和模式3（文字OCR）
"""
import os
import cv2
import numpy as np
import shutil
import json
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QListWidget, QPushButton, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QCheckBox, QInputDialog
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


def cv_imread_png(filepath: str):
    """读取PNG图片，保留alpha通道"""
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def convert_rgba_to_rgb(img: np.ndarray, bg_color=(255, 255, 255)):
    """将RGBA图片转为RGB，透明部分填充背景色"""
    if img is None:
        return None
    if len(img.shape) == 2:
        # 灰度图
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        # 有alpha通道
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3]
        bg = np.full_like(rgb, bg_color, dtype=np.uint8)
        # 混合
        result = (rgb * alpha[:, :, np.newaxis] + bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        return result
    return img


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


class SiameseFolderWidget(QWidget):
    """分离式分类标注组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.data_folder: str = ""
        self.output_folder: str = ""
        self.samples: List[Dict] = []
        self.current_sample_idx: int = -1
        
        # YOLO 检测器
        self.detector = None
        
        # 当前数据
        self.cropped_images: List[np.ndarray] = []
        self.hint_images: List[Dict] = []
        
        # 选择状态
        self.selected_crop_idx: int = -1
        
        # 类别统计
        self.class_counts: Dict[str, int] = {}
        
        # 已完成的样本索引
        self.completed_samples: set = set()
        
        # OCR模式
        self.ocr_enabled = False
        self.ocr_engine = None
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 顶部工具栏
        toolbar = QHBoxLayout()
        
        self.load_model_btn = QPushButton("加载YOLO模型")
        self.load_model_btn.clicked.connect(self._load_model)
        toolbar.addWidget(self.load_model_btn)
        
        self.load_data_btn = QPushButton("加载数据文件夹")
        self.load_data_btn.clicked.connect(self._load_data_folder)
        toolbar.addWidget(self.load_data_btn)
        
        # OCR 模式
        self.ocr_checkbox = QCheckBox("启用OCR识别")
        self.ocr_checkbox.stateChanged.connect(self._on_ocr_changed)
        toolbar.addWidget(self.ocr_checkbox)
        
        self.model_label = QLabel("未加载模型")
        toolbar.addWidget(self.model_label)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # 主内容区
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.bg_label = QLabel("加载数据文件夹后显示背景图")
        self.bg_label.setMinimumSize(300, 150)
        self.bg_label.setMaximumHeight(250)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        left_layout.addWidget(self.bg_label, 1)
        
        # 检测到的目标
        crop_group = QGroupBox("检测到的目标（点击选择）")
        crop_layout = QHBoxLayout(crop_group)
        self.crop_scroll = QScrollArea()
        self.crop_scroll.setWidgetResizable(True)
        self.crop_container = QWidget()
        self.crop_grid = QHBoxLayout(self.crop_container)
        self.crop_grid.setAlignment(Qt.AlignLeft)
        self.crop_scroll.setWidget(self.crop_container)
        self.crop_scroll.setMinimumHeight(120)
        self.crop_scroll.setMaximumHeight(200)
        crop_layout.addWidget(self.crop_scroll)
        left_layout.addWidget(crop_group, 1)
        
        # 提示图
        hint_group = QGroupBox("提示图（点击匹配，文件名=类别名）")
        hint_layout = QHBoxLayout(hint_group)
        self.hint_scroll = QScrollArea()
        self.hint_scroll.setWidgetResizable(True)
        self.hint_container = QWidget()
        self.hint_grid = QHBoxLayout(self.hint_container)
        self.hint_grid.setAlignment(Qt.AlignLeft)
        self.hint_scroll.setWidget(self.hint_container)
        self.hint_scroll.setMinimumHeight(120)
        self.hint_scroll.setMaximumHeight(200)
        hint_layout.addWidget(self.hint_scroll)
        left_layout.addWidget(hint_group, 1)
        
        splitter.addWidget(left_widget)
        
        # 右侧
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        sample_group = QGroupBox("样本列表")
        sample_layout = QVBoxLayout(sample_group)
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._on_sample_selected)
        sample_layout.addWidget(self.sample_list)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一个")
        self.prev_btn.clicked.connect(self._prev_sample)
        self.next_btn = QPushButton("下一个")
        self.next_btn.clicked.connect(self._next_sample)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        sample_layout.addLayout(nav_layout)
        
        right_layout.addWidget(sample_group)
        
        stats_group = QGroupBox("类别统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_list = QListWidget()
        stats_layout.addWidget(self.stats_list)
        right_layout.addWidget(stats_group)
        
        self.skip_btn = QPushButton("跳过当前样本")
        self.skip_btn.clicked.connect(self._next_sample)
        right_layout.addWidget(self.skip_btn)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 250])
    
    def _load_model(self):
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
        folder = QFileDialog.getExistingDirectory(self, "选择数据文件夹")
        if not folder:
            return
        
        self.data_folder = folder
        self.samples = []
        
        for item in sorted(os.listdir(folder)):
            item_path = os.path.join(folder, item)
            if not os.path.isdir(item_path):
                continue
            
            # 查找 bg 图片
            bg_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for name in ['bg', 'background', '背景']:
                    test_path = os.path.join(item_path, name + ext)
                    if os.path.exists(test_path):
                        bg_path = test_path
                        break
                if bg_path:
                    break
            
            if not bg_path:
                for f in os.listdir(item_path):
                    if f.lower().endswith(('.jpg', '.jpeg')):
                        bg_path = os.path.join(item_path, f)
                        break
            
            if not bg_path:
                continue
            
            # 查找 icon 图片
            icons = []
            for f in sorted(os.listdir(item_path)):
                if f.lower().endswith('.png'):
                    icon_name = os.path.splitext(f)[0]
                    icons.append({
                        'name': icon_name,
                        'path': os.path.join(item_path, f)
                    })
            
            if icons:
                self.samples.append({
                    'name': item,
                    'path': item_path,
                    'bg': bg_path,
                    'icons': icons
                })
        
        self.sample_list.clear()
        for i, sample in enumerate(self.samples):
            name = sample['name']
            if i in self.completed_samples:
                name = f"✓ {name}"
            self.sample_list.addItem(name)
        
        # 输出目录 = 数据文件夹/datasets
        self.output_folder = os.path.join(folder, "datasets")
        
        # 尝试加载进度
        self._load_progress()
        
        # 跳到第一个未完成的样本
        for i in range(len(self.samples)):
            if i not in self.completed_samples:
                self.sample_list.setCurrentRow(i)
                break
        else:
            if self.samples:
                self.sample_list.setCurrentRow(0)
    
    def _save_progress(self):
        """保存标注进度（自动保存，不弹窗）"""
        if not self.data_folder:
            return
        
        progress_file = os.path.join(self.data_folder, ".siamese_progress.json")
        data = {
            'completed_samples': list(self.completed_samples),
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
        
        progress_file = os.path.join(self.data_folder, ".siamese_progress.json")
        if not os.path.exists(progress_file):
            return
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.completed_samples = set(data.get('completed_samples', []))
            self.class_counts = data.get('class_counts', {})
            
            # 更新列表显示
            for i in range(self.sample_list.count()):
                if i in self.completed_samples:
                    item = self.sample_list.item(i)
                    if item and not item.text().startswith('✓'):
                        item.setText(f"✓ {self.samples[i]['name']}")
            
            self._update_stats()
        except Exception as e:
            print(f"加载进度失败: {e}")
    
    def _on_ocr_changed(self, state):
        self.ocr_enabled = state == Qt.Checked
        if self.ocr_enabled and self.ocr_engine is None:
            try:
                import ddddocr
                self.ocr_engine = ddddocr.DdddOcr()
            except ImportError:
                QMessageBox.warning(self, "警告", "未安装 ddddocr，请运行: pip install ddddocr")
                self.ocr_checkbox.setChecked(False)
                self.ocr_enabled = False
    
    def _on_sample_selected(self, row: int):
        if row < 0 or row >= len(self.samples):
            return
        
        self.current_sample_idx = row
        sample = self.samples[row]
        
        bg_img = cv_imread(sample['bg'])
        if bg_img is None:
            return
        
        # 检测
        self.cropped_images = []
        if self.detector:
            try:
                detections = self.detector.detect(sample['bg'])
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = bg_img[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        self.cropped_images.append(crop)
            except Exception as e:
                print(f"检测错误: {e}")
        
        self._display_bg(bg_img)
        self._display_crops()
        
        # 加载提示图
        self.hint_images = []
        for icon in sample['icons']:
            # 使用UNCHANGED读取以保留alpha通道
            img = cv_imread_png(icon['path'])
            if img is not None:
                name = icon['name']
                # OCR 识别
                if self.ocr_enabled and self.ocr_engine:
                    try:
                        # 转为RGB用于OCR
                        img_rgb = convert_rgba_to_rgb(img)
                        _, buf = cv2.imencode('.jpg', img_rgb)
                        result = self.ocr_engine.classification(buf.tobytes())
                        if result:
                            name = result
                    except:
                        pass
                
                self.hint_images.append({
                    'name': name,
                    'image': img,
                    'path': icon['path']
                })
        
        self._display_hints()
        self.selected_crop_idx = -1
    
    def _display_bg(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.bg_label.width() - 10, self.bg_label.height() - 10,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.bg_label.setPixmap(scaled)
    
    def _display_crops(self):
        while self.crop_grid.count():
            item = self.crop_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, crop in enumerate(self.cropped_images):
            label = ClickableImageLabel(i)
            label.clicked.connect(self._on_crop_clicked)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            self.crop_grid.addWidget(label)
    
    def _display_hints(self):
        while self.hint_grid.count():
            item = self.hint_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, hint in enumerate(self.hint_images):
            label = ClickableImageLabel(i)
            label.clicked.connect(self._on_hint_clicked)
            
            # 处理alpha通道
            img = hint['image']
            rgb = convert_rgba_to_rgb(img)
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            label.setToolTip(hint['name'])
            self.hint_grid.addWidget(label)
    
    def _on_crop_clicked(self, idx):
        self.selected_crop_idx = idx
        for i in range(self.crop_grid.count()):
            widget = self.crop_grid.itemAt(i).widget()
            if isinstance(widget, ClickableImageLabel):
                widget.set_selected(i == idx)
    
    def _on_hint_clicked(self, idx):
        if self.selected_crop_idx < 0 or self.selected_crop_idx >= len(self.cropped_images):
            QMessageBox.warning(self, "提示", "请先选择一个检测到的目标")
            return
        
        crop_img = self.cropped_images[self.selected_crop_idx]
        hint = self.hint_images[idx]
        default_name = hint['name']
        
        # 弹出确认对话框，允许修改类别名
        class_name, ok = QInputDialog.getText(
            self, "确认分类",
            "请确认或修改类别名称:",
            text=default_name
        )
        
        if not ok or not class_name.strip():
            return
        
        class_name = class_name.strip()
        
        self._save_crop(crop_img, class_name, hint['path'])
        
        del self.cropped_images[self.selected_crop_idx]
        self.selected_crop_idx = -1
        self._display_crops()
        self._update_stats()
        
        if not self.cropped_images:
            # 标记为已完成
            self.completed_samples.add(self.current_sample_idx)
            item = self.sample_list.item(self.current_sample_idx)
            if item:
                item.setText(f"✓ {self.samples[self.current_sample_idx]['name']}")
            # 自动保存进度
            self._save_progress()
            self._next_sample()
    
    def _save_crop(self, crop_img, class_name, hint_path):
        # 使用 datasets 子目录
        if not self.output_folder:
            self.output_folder = os.path.join(self.data_folder, "datasets")
        
        class_folder = os.path.join(self.output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        existing = len([f for f in os.listdir(class_folder) if f.startswith('crop_')])
        crop_path = os.path.join(class_folder, f"crop_{existing + 1:04d}.jpg")
        cv_imwrite(crop_path, crop_img)
        
        hint_dst = os.path.join(class_folder, "hint.png")
        if not os.path.exists(hint_dst):
            shutil.copy2(hint_path, hint_dst)
        
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
    
    def _update_stats(self):
        self.stats_list.clear()
        for name, count in sorted(self.class_counts.items()):
            self.stats_list.addItem(f"{name}: {count} 张")
    
    def _prev_sample(self):
        if self.current_sample_idx > 0:
            self.sample_list.setCurrentRow(self.current_sample_idx - 1)
    
    def _next_sample(self):
        if self.current_sample_idx < len(self.samples) - 1:
            self.sample_list.setCurrentRow(self.current_sample_idx + 1)
