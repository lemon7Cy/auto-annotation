# -*- coding: utf-8 -*-
"""
半自动图像标注工具主窗口
支持项目管理、最近项目、自动预测
"""
import os
import cv2
import numpy as np
from typing import Optional, Any
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QPushButton, QToolBar,
    QAction, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QStatusBar, QInputDialog,
    QMenu, QMenuBar, QCheckBox, QDialog, QLineEdit, QDialogButtonBox,
    QListView, QAbstractItemView, QTabWidget
)
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtCore import Qt, QSize

from image_canvas import ImageCanvas
from annotation_utils import save_annotations, load_annotations, get_label_path
from project_manager import Project, RecentProjectsManager
from siamese_widget import SiameseWidget


def cv_imread(filepath: str):
    """读取图片，支持中文路径"""
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)


class NewProjectDialog(QDialog):
    """新建项目对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # 项目名称
        name_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: 手势识别训练")
        name_layout.addRow("项目名称:", self.name_edit)
        layout.addLayout(name_layout)
        
        # 项目位置
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择项目保存位置...")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_btn)
        
        path_form = QFormLayout()
        path_form.addRow("保存位置:", path_layout)
        layout.addLayout(path_form)
        
        # 初始类别
        classes_layout = QFormLayout()
        self.classes_edit = QLineEdit()
        self.classes_edit.setText("object")
        self.classes_edit.setPlaceholderText("多个类别用逗号分隔，如: cat,dog,bird")
        classes_layout.addRow("初始类别:", self.classes_edit)
        layout.addLayout(classes_layout)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择项目保存位置")
        if path:
            self.path_edit.setText(path)
    
    def get_project_info(self):
        """获取项目信息"""
        name = self.name_edit.text().strip()
        base_path = self.path_edit.text().strip()
        classes = [c.strip() for c in self.classes_edit.text().split(',') if c.strip()]
        
        if not name or not base_path:
            return None
        
        return {
            'name': name,
            'path': os.path.join(base_path, name),
            'classes': classes or ['object']
        }


class MainWindow(QMainWindow):
    """半自动标注工具主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 项目管理
        self.project: Optional[Project] = None
        self.recent_manager = RecentProjectsManager()
        
        # 状态
        self.image_files: list = []
        self.current_idx: int = -1
        self.detector: Optional[Any] = None
        self.unsaved_changes: bool = False
        
        # 类别尺寸记忆 {class_id: (width, height)}
        self.class_sizes: dict = {}
        
        # 初始化UI
        self._init_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        self._connect_signals()
        
        self.setWindowTitle("多功能图像标注工具")
        self.resize(1200, 850)  # 窗口大小，可调整
        
        # 更新最近项目菜单
        self._update_recent_menu()
    
    def _init_ui(self):
        """初始化UI布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab切换
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab1: YOLO标注
        yolo_tab = QWidget()
        layout = QHBoxLayout(yolo_tab)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：图像画布
        self.canvas = ImageCanvas()
        splitter.addWidget(self.canvas)
        
        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        
        # 项目信息
        project_group = QGroupBox("项目信息")
        project_layout = QFormLayout(project_group)
        self.project_name_label = QLabel("未打开项目")
        project_layout.addRow("项目:", self.project_name_label)
        self.image_count_label = QLabel("--")
        project_layout.addRow("图片:", self.image_count_label)
        right_layout.addWidget(project_group)
        
        # 图像列表
        list_group = QGroupBox("图像列表")
        list_layout = QVBoxLayout(list_group)
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(250)
        list_layout.addWidget(self.image_list)
        right_layout.addWidget(list_group)
        
        # 检测设置
        detect_group = QGroupBox("检测设置")
        detect_layout = QFormLayout(detect_group)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        detect_layout.addRow("置信度阈值:", self.conf_spin)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        detect_layout.addRow("NMS IoU阈值:", self.iou_spin)
        
        self.detect_btn = QPushButton("运行检测")
        self.detect_btn.setEnabled(False)
        detect_layout.addRow(self.detect_btn)
        
        # 自动预测勾选框
        self.auto_predict_checkbox = QCheckBox("切换图片自动预测")
        self.auto_predict_checkbox.setEnabled(False)
        detect_layout.addRow(self.auto_predict_checkbox)
        
        right_layout.addWidget(detect_group)
        
        # BBox信息
        bbox_group = QGroupBox("选中框信息")
        bbox_layout = QFormLayout(bbox_group)
        
        self.class_combo = QComboBox()
        self.class_combo.setEnabled(False)
        bbox_layout.addRow("类别:", self.class_combo)
        
        self.confidence_label = QLabel("--")
        bbox_layout.addRow("置信度:", self.confidence_label)
        
        self.coords_label = QLabel("--")
        bbox_layout.addRow("坐标:", self.coords_label)
        
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.setEnabled(False)
        bbox_layout.addRow(self.delete_btn)
        
        right_layout.addWidget(bbox_group)
        
        # 类别管理
        class_group = QGroupBox("类别管理")
        class_layout = QVBoxLayout(class_group)
        
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(100)
        class_layout.addWidget(self.class_list)
        
        btn_layout = QHBoxLayout()
        self.add_class_btn = QPushButton("添加")
        self.remove_class_btn = QPushButton("删除")
        btn_layout.addWidget(self.add_class_btn)
        btn_layout.addWidget(self.remove_class_btn)
        class_layout.addLayout(btn_layout)
        
        right_layout.addWidget(class_group)
        
        # 当前绘制类别
        draw_group = QGroupBox("绘制设置")
        draw_layout = QFormLayout(draw_group)
        
        self.draw_class_combo = QComboBox()
        draw_layout.addRow("绘制类别:", self.draw_class_combo)
        
        # 点击放置模式
        self.click_mode_checkbox = QCheckBox("点击放置模式")
        self.click_mode_checkbox.setToolTip("勾选后，点击中心点直接生成指定大小的框")
        draw_layout.addRow(self.click_mode_checkbox)
        
        # 固定框尺寸
        self.bbox_width_spin = QSpinBox()
        self.bbox_width_spin.setRange(10, 500)
        self.bbox_width_spin.setValue(80)
        draw_layout.addRow("框宽度:", self.bbox_width_spin)
        
        self.bbox_height_spin = QSpinBox()
        self.bbox_height_spin.setRange(10, 500)
        self.bbox_height_spin.setValue(80)
        draw_layout.addRow("框高度:", self.bbox_height_spin)
        
        right_layout.addWidget(draw_group)
        
        right_layout.addStretch()
        
        # 设置分割比例
        splitter.setSizes([1000, 350])
        
        # 添加 YOLO 标注 Tab
        self.tab_widget.addTab(yolo_tab, "YOLO 标注")
        
        # Tab2: 孪生分类
        self.siamese_widget = SiameseWidget()
        self.tab_widget.addTab(self.siamese_widget, "孪生分类")
    
    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        new_project_action = QAction("新建项目(&N)...", self)
        new_project_action.setShortcut(QKeySequence.New)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("打开项目(&O)...", self)
        open_project_action.setShortcut(QKeySequence.Open)
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)
        
        # 最近项目子菜单
        self.recent_menu = file_menu.addMenu("最近项目(&R)")
        
        file_menu.addSeparator()
        
        add_images_action = QAction("添加图片文件夹(&I)...", self)
        add_images_action.triggered.connect(self._add_images)
        file_menu.addAction(add_images_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("保存标注(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_annotations)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 模型菜单
        model_menu = menubar.addMenu("模型(&M)")
        
        load_model_action = QAction("加载ONNX模型(&L)...", self)
        load_model_action.triggered.connect(self._load_model)
        model_menu.addAction(load_model_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        
        # 新建项目
        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self._new_project)
        toolbar.addAction(new_action)
        
        # 打开项目
        open_action = QAction("打开项目", self)
        open_action.triggered.connect(self._open_project)
        toolbar.addAction(open_action)
        
        # 添加图片
        add_img_action = QAction("添加图片文件夹", self)
        add_img_action.triggered.connect(self._add_images)
        toolbar.addAction(add_img_action)
        
        toolbar.addSeparator()
        
        # 加载模型
        model_action = QAction("加载模型", self)
        model_action.triggered.connect(self._load_model)
        toolbar.addAction(model_action)
        
        toolbar.addSeparator()
        
        # 保存
        save_action = QAction("保存", self)
        save_action.triggered.connect(self._save_annotations)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 导航（使用左右方向键）
        prev_action = QAction("上一张", self)
        prev_action.setShortcut(Qt.Key_Left)
        prev_action.triggered.connect(self._prev_image)
        toolbar.addAction(prev_action)
        
        next_action = QAction("下一张", self)
        next_action.setShortcut(Qt.Key_Right)
        next_action.triggered.connect(self._next_image)
        toolbar.addAction(next_action)
    
    def _create_statusbar(self):
        """创建状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_image = QLabel("未加载图像")
        self.status_bbox = QLabel("标注数: 0")
        self.status_model = QLabel("未加载模型")
        
        self.statusbar.addWidget(self.status_image, 1)
        self.statusbar.addWidget(self.status_bbox)
        self.statusbar.addWidget(self.status_model)
    
    def _connect_signals(self):
        """连接信号"""
        # 图像列表
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        
        # Canvas信号
        self.canvas.bbox_selected.connect(self._on_bbox_selected)
        self.canvas.annotations_changed.connect(self._on_annotations_changed)
        
        # 检测按钮
        self.detect_btn.clicked.connect(self._run_detection)
        
        # 删除按钮
        self.delete_btn.clicked.connect(self.canvas.delete_selected)
        
        # 类别下拉框
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.draw_class_combo.currentIndexChanged.connect(self._on_draw_class_changed)
        
        # 类别管理
        self.add_class_btn.clicked.connect(self._add_class)
        self.remove_class_btn.clicked.connect(self._remove_class)
        
        # 点击放置模式
        self.click_mode_checkbox.stateChanged.connect(self._on_click_mode_changed)
        self.bbox_width_spin.valueChanged.connect(self._on_bbox_size_changed)
        self.bbox_height_spin.valueChanged.connect(self._on_bbox_size_changed)
    
    def _on_click_mode_changed(self, state):
        """点击放置模式切换"""
        enabled = state == Qt.Checked
        self.canvas.set_click_mode(
            enabled,
            self.bbox_width_spin.value(),
            self.bbox_height_spin.value()
        )
    
    def _on_bbox_size_changed(self, value):
        """框尺寸变化 - 保存到当前类别"""
        current_class = self.draw_class_combo.currentIndex()
        width = self.bbox_width_spin.value()
        height = self.bbox_height_spin.value()
        
        # 保存到类别尺寸记忆
        self.class_sizes[current_class] = (width, height)
        
        # 更新canvas
        self.canvas.set_click_mode(
            self.click_mode_checkbox.isChecked(),
            width,
            height
        )
    
    def _update_recent_menu(self):
        """更新最近项目菜单"""
        self.recent_menu.clear()
        
        recent = self.recent_manager.get_recent_projects()
        if not recent:
            action = QAction("(无最近项目)", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        
        for p in recent:
            action = QAction(f"{p['name']} - {p['path']}", self)
            action.setData(p['path'])
            action.triggered.connect(lambda checked, path=p['path']: self._open_project_path(path))
            self.recent_menu.addAction(action)
        
        self.recent_menu.addSeparator()
        clear_action = QAction("清除最近项目", self)
        clear_action.triggered.connect(self._clear_recent)
        self.recent_menu.addAction(clear_action)
    
    def _clear_recent(self):
        """清除最近项目"""
        self.recent_manager.clear()
        self._update_recent_menu()
    
    def _new_project(self):
        """新建项目"""
        if self.unsaved_changes:
            self._save_annotations()
        
        dialog = NewProjectDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        
        info = dialog.get_project_info()
        if not info:
            QMessageBox.warning(self, "警告", "请填写项目名称和保存位置")
            return
        
        # 检查目录是否已存在
        if os.path.exists(info['path']):
            reply = QMessageBox.question(
                self, "目录已存在",
                f"目录 {info['path']} 已存在，是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # 创建项目
        self.project = Project(info['path'])
        self.project.create(info['classes'])
        
        # 添加到最近项目
        self.recent_manager.add_project(info['path'], info['name'])
        self._update_recent_menu()
        
        # 加载项目
        self._load_project_ui()
        
        self.statusbar.showMessage(f"已创建项目: {info['name']}")
    
    def _open_project(self):
        """打开项目"""
        if self.unsaved_changes:
            self._save_annotations()
        
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self._open_project_path(path)
    
    def _open_project_path(self, path: str):
        """打开指定路径的项目"""
        if self.unsaved_changes:
            self._save_annotations()
        
        self.project = Project(path)
        if not self.project.load():
            # 尝试作为普通目录打开（向后兼容）
            if os.path.isdir(path):
                self.project.create()
        
        # 添加到最近项目
        self.recent_manager.add_project(path, self.project.name)
        self._update_recent_menu()
        
        # 加载项目UI
        self._load_project_ui()
        
        self.statusbar.showMessage(f"已打开项目: {self.project.name}")
    
    def _load_project_ui(self):
        """加载项目到UI"""
        if not self.project:
            return
        
        # 更新项目信息
        self.project_name_label.setText(self.project.name)
        
        # 更新类别列表
        self._update_class_lists()
        
        # 加载图像列表
        self.image_files = self.project.get_all_images()
        self.image_list.clear()
        
        labeled_count = 0
        for img_path in self.image_files:
            basename = os.path.basename(img_path)
            item = QListWidgetItem(basename)
            if self.project.is_image_labeled(img_path):
                item.setText(f"✓ {basename}")
                labeled_count += 1
            self.image_list.addItem(item)
        
        # 更新图片计数
        total = len(self.image_files)
        self.image_count_label.setText(f"{labeled_count}/{total} 已标注")
        
        # 自动选择第一张图片
        if self.image_files:
            self.image_list.setCurrentRow(0)
        
        self.setWindowTitle(f"YOLOv5 半自动标注工具 - {self.project.name}")
    
    def _add_images(self):
        """添加图片文件夹到项目"""
        if not self.project:
            QMessageBox.warning(self, "警告", "请先创建或打开项目")
            return
        
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        
        if not folder:
            return
        
        # 获取文件夹中的所有图片
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(extensions):
                files.append(os.path.join(folder, filename))
        
        if not files:
            QMessageBox.information(self, "提示", "该文件夹中没有找到图片文件")
            return
        
        added = self.project.add_images(files)
        
        # 刷新图像列表
        self._load_project_ui()
        
        self.statusbar.showMessage(f"已添加 {added} 张图片")
    
    def _load_model(self):
        """加载ONNX模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择ONNX模型", "", "ONNX模型 (*.onnx)"
        )
        if not model_path:
            return
        
        try:
            # Lazy import to avoid startup errors
            from yolo_detector import YOLODetector
            
            class_names = self.project.classes if self.project else ['object']
            
            self.detector = YOLODetector(
                model_path,
                conf_thres=self.conf_spin.value(),
                iou_thres=self.iou_spin.value(),
                class_names=class_names
            )
            self.detect_btn.setEnabled(True)
            self.auto_predict_checkbox.setEnabled(True)
            self.status_model.setText(f"模型: {os.path.basename(model_path)}")
            self.statusbar.showMessage("模型加载成功")
            
            # 保存模型路径到项目
            if self.project:
                self.project.model_path = model_path
                self.project.save()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
    
    def _on_image_selected(self, row: int):
        """图像选择变化"""
        if self.unsaved_changes:
            self._save_annotations()
        
        if row < 0 or row >= len(self.image_files):
            return
        
        self.current_idx = row
        img_path = self.image_files[row]
        
        # 加载图像
        img = cv_imread(img_path)
        if img is None:
            QMessageBox.warning(self, "警告", f"无法加载图像: {img_path}")
            return
        
        self.canvas.set_image(img)
        if self.project:
            self.canvas.set_class_names(self.project.classes)
        
        # 加载已有标注
        labels_dir = self.project.labels_dir if self.project else None
        label_path = get_label_path(img_path, labels_dir)
        if os.path.exists(label_path):
            h, w = img.shape[:2]
            bboxes = load_annotations(label_path, w, h)
            self.canvas.set_bboxes(bboxes)
        else:
            self.canvas.set_bboxes([])
        
        self.unsaved_changes = False
        self.status_image.setText(f"图像: {os.path.basename(img_path)} ({row + 1}/{len(self.image_files)})")
        self._update_bbox_info()
        
        # 自动预测：如果勾选了"切换图片自动预测"且图片尚未标注
        if self.auto_predict_checkbox.isChecked() and self.detector is not None:
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                self._run_detection()
    
    def _run_detection(self):
        """运行YOLO检测"""
        if self.detector is None or self.current_idx < 0:
            return
        
        # 更新阈值
        self.detector.set_thresholds(
            conf_thres=self.conf_spin.value(),
            iou_thres=self.iou_spin.value()
        )
        if self.project:
            self.detector.set_class_names(self.project.classes)
        
        img_path = self.image_files[self.current_idx]
        
        try:
            detections = self.detector.detect(img_path)
            self.canvas.set_bboxes(detections)
            self.unsaved_changes = True
            self._update_bbox_info()
            self.statusbar.showMessage(f"检测到 {len(detections)} 个目标")
        except Exception as e:
            QMessageBox.warning(self, "检测错误", str(e))
    
    def _save_annotations(self):
        """保存当前标注"""
        if self.current_idx < 0 or not self.project:
            return
        
        img_path = self.image_files[self.current_idx]
        label_path = get_label_path(img_path, self.project.labels_dir)
        
        img = cv_imread(img_path)
        if img is None:
            return
        h, w = img.shape[:2]
        
        bboxes = self.canvas.get_bboxes()
        save_annotations(label_path, bboxes, w, h)
        
        # 标记图片为已标注（复制到images目录）
        if bboxes:
            self.project.mark_as_labeled(img_path)
        
        # 更新列表项显示
        basename = os.path.basename(img_path)
        if bboxes:
            self.image_list.item(self.current_idx).setText(f"✓ {basename}")
        else:
            self.image_list.item(self.current_idx).setText(basename)
        
        self.unsaved_changes = False
        self._update_bbox_info()
        
        # 更新图片计数
        self._update_image_count()
        
        self.statusbar.showMessage("标注已保存")
    
    def _update_image_count(self):
        """更新图片计数显示"""
        if not self.project:
            return
        
        labeled = sum(1 for img in self.image_files if self.project.is_image_labeled(img))
        total = len(self.image_files)
        self.image_count_label.setText(f"{labeled}/{total} 已标注")
    
    def _prev_image(self):
        """上一张图像"""
        if self.current_idx > 0:
            self.image_list.setCurrentRow(self.current_idx - 1)
    
    def _next_image(self):
        """下一张图像"""
        if self.current_idx < len(self.image_files) - 1:
            self.image_list.setCurrentRow(self.current_idx + 1)
    
    def _on_bbox_selected(self, idx: int):
        """BBox选中事件"""
        self._update_bbox_info(idx)
        self.delete_btn.setEnabled(idx >= 0)
        self.class_combo.setEnabled(idx >= 0)
    
    def _on_annotations_changed(self):
        """标注变化事件"""
        self.unsaved_changes = True
        self._update_bbox_info()
    
    def _update_bbox_info(self, selected_idx: int = -1):
        """更新BBox信息显示"""
        bboxes = self.canvas.get_bboxes()
        self.status_bbox.setText(f"标注数: {len(bboxes)}")
        
        if selected_idx >= 0 and selected_idx < len(bboxes):
            bbox = bboxes[selected_idx]
            self.class_combo.setCurrentIndex(bbox.get('class_id', 0))
            self.confidence_label.setText(f"{bbox.get('confidence', 1.0):.2f}")
            coords = bbox.get('bbox', [0, 0, 0, 0])
            self.coords_label.setText(f"[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]")
        else:
            self.confidence_label.setText("--")
            self.coords_label.setText("--")
    
    def _on_class_changed(self, idx: int):
        """修改选中BBox的类别"""
        if idx >= 0 and self.canvas.selected_idx >= 0:
            self.canvas.change_selected_class(idx)
    
    def _on_draw_class_changed(self, idx: int):
        """修改绘制类别 - 加载该类别记忆的尺寸"""
        if idx >= 0:
            self.canvas.set_current_class(idx)
            
            # 加载该类别保存的尺寸
            if idx in self.class_sizes:
                width, height = self.class_sizes[idx]
                # 临时断开信号避免触发保存
                self.bbox_width_spin.blockSignals(True)
                self.bbox_height_spin.blockSignals(True)
                self.bbox_width_spin.setValue(width)
                self.bbox_height_spin.setValue(height)
                self.bbox_width_spin.blockSignals(False)
                self.bbox_height_spin.blockSignals(False)
                
                # 更新canvas
                self.canvas.set_click_mode(
                    self.click_mode_checkbox.isChecked(),
                    width,
                    height
                )
    
    def _update_class_lists(self):
        """更新类别列表显示"""
        class_names = self.project.classes if self.project else ['object']
        
        # 类别列表
        self.class_list.clear()
        for name in class_names:
            self.class_list.addItem(name)
        
        # 下拉框
        self.class_combo.clear()
        self.class_combo.addItems(class_names)
        
        self.draw_class_combo.clear()
        self.draw_class_combo.addItems(class_names)
    
    def _add_class(self):
        """添加新类别"""
        if not self.project:
            return
        
        name, ok = QInputDialog.getText(self, "添加类别", "类别名称:")
        if ok and name.strip():
            self.project.classes.append(name.strip())
            self._update_class_lists()
            self.project.save()
    
    def _remove_class(self):
        """删除选中类别"""
        if not self.project:
            return
        
        row = self.class_list.currentRow()
        if row >= 0 and len(self.project.classes) > 1:
            del self.project.classes[row]
            self._update_class_lists()
            self.project.save()
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于",
            "YOLOv5 半自动标注工具\n\n"
            "基于 ONNX Runtime 的图像标注工具\n"
            "无 PyTorch 依赖\n\n"
            "快捷键:\n"
            "  ← → - 上一张/下一张\n"
            "  Delete - 删除选中框\n"
            "  Ctrl+S - 保存\n"
            "  Ctrl+N - 新建项目\n"
            "  Ctrl+O - 打开项目"
        )
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "当前有未保存的标注，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self._save_annotations()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
