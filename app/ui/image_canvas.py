# -*- coding: utf-8 -*-
"""
图像画布组件
支持图像显示、BBox可视化、交互式标注编辑
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from PyQt5.QtWidgets import QWidget, QMenu, QAction, QInputDialog
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QFont, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint


# 预定义的颜色列表（用于不同类别）
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
    (0, 128, 255),    # 天蓝
    (128, 255, 0),    # 柠檬绿
]


class ImageCanvas(QWidget):
    """图像画布：支持BBox显示、绘制、编辑"""
    
    # 信号
    bbox_selected = pyqtSignal(int)       # 选中BBox时发射，参数为索引
    bbox_created = pyqtSignal(dict)       # 新建BBox时发射
    bbox_modified = pyqtSignal(int)       # 修改BBox时发射
    bbox_deleted = pyqtSignal(int)        # 删除BBox时发射
    annotations_changed = pyqtSignal()    # 标注发生任何变化时发射
    
    # 编辑模式
    MODE_VIEW = 0      # 查看模式
    MODE_DRAW = 1      # 绘制新BBox模式
    MODE_EDIT = 2      # 编辑已有BBox模式
    
    # 调整手柄位置
    HANDLE_NONE = -1
    HANDLE_TL = 0   # 左上
    HANDLE_TR = 1   # 右上
    HANDLE_BR = 2   # 右下
    HANDLE_BL = 3   # 左下
    HANDLE_CENTER = 4  # 中心（移动）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 图像数据
        self.image: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None
        self.img_w = 0
        self.img_h = 0
        
        # 标注数据
        self.bboxes: List[Dict] = []
        self.selected_idx = -1
        self.current_class_id = 0
        self.class_names: List[str] = []
        
        # 显示参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 交互状态
        self.mode = self.MODE_DRAW  # 默认绘制模式
        self.drawing = False
        self.resizing = False
        self.moving = False
        self.active_handle = self.HANDLE_NONE
        
        # 临时绘制状态
        self.draw_start = None
        self.draw_end = None
        self.drag_offset = (0, 0)
        
        # 手柄大小
        self.handle_size = 8
        
        # 点击放置模式
        self.click_mode = False
        self.fixed_width = 80
        self.fixed_height = 80
        
        # 设置焦点策略以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        
        # 最小尺寸
        self.setMinimumSize(400, 300)
    
    def set_image(self, image: np.ndarray):
        """设置当前图像"""
        self.image = image.copy()
        self.img_h, self.img_w = image.shape[:2]
        
        # 转换为QPixmap
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        
        # 计算缩放和偏移以居中显示
        self._update_transform()
        
        # 清除选择
        self.selected_idx = -1
        self.bboxes = []
        
        self.update()
    
    def set_bboxes(self, bboxes: List[Dict]):
        """设置BBox列表"""
        self.bboxes = [b.copy() for b in bboxes]
        self.selected_idx = -1
        self.update()
    
    def get_bboxes(self) -> List[Dict]:
        """获取当前BBox列表"""
        return [b.copy() for b in self.bboxes]
    
    def set_class_names(self, class_names: List[str]):
        """设置类别名称列表"""
        self.class_names = class_names
    
    def set_current_class(self, class_id: int):
        """设置当前绘制类别"""
        self.current_class_id = class_id
    
    def set_mode(self, mode: int):
        """设置编辑模式"""
        self.mode = mode
        self.update()
    
    def set_click_mode(self, enabled: bool, width: int = 80, height: int = 80):
        """设置点击放置模式"""
        self.click_mode = enabled
        self.fixed_width = width
        self.fixed_height = height
    
    def delete_selected(self):
        """删除选中的BBox"""
        if 0 <= self.selected_idx < len(self.bboxes):
            del self.bboxes[self.selected_idx]
            self.bbox_deleted.emit(self.selected_idx)
            self.selected_idx = -1
            self.annotations_changed.emit()
            self.update()
    
    def change_selected_class(self, class_id: int):
        """修改选中BBox的类别"""
        if 0 <= self.selected_idx < len(self.bboxes):
            self.bboxes[self.selected_idx]['class_id'] = class_id
            self.bbox_modified.emit(self.selected_idx)
            self.annotations_changed.emit()
            self.update()
    
    def _update_transform(self):
        """更新缩放和偏移参数"""
        if self.pixmap is None:
            return
        
        # 计算适应窗口的缩放比例
        w_scale = self.width() / self.img_w
        h_scale = self.height() / self.img_h
        self.scale = min(w_scale, h_scale, 1.0)  # 不放大，只缩小
        
        # 计算居中偏移
        scaled_w = self.img_w * self.scale
        scaled_h = self.img_h * self.scale
        self.offset_x = (self.width() - scaled_w) / 2
        self.offset_y = (self.height() - scaled_h) / 2
    
    def _img_to_widget(self, x: float, y: float) -> Tuple[int, int]:
        """图像坐标转窗口坐标"""
        wx = int(x * self.scale + self.offset_x)
        wy = int(y * self.scale + self.offset_y)
        return wx, wy
    
    def _widget_to_img(self, wx: int, wy: int) -> Tuple[int, int]:
        """窗口坐标转图像坐标"""
        x = int((wx - self.offset_x) / self.scale)
        y = int((wy - self.offset_y) / self.scale)
        # 裁剪到图像边界
        x = max(0, min(self.img_w, x))
        y = max(0, min(self.img_h, y))
        return x, y
    
    def _get_handle_at(self, wx: int, wy: int, bbox_idx: int) -> int:
        """检查窗口坐标是否在BBox的调整手柄上"""
        if bbox_idx < 0 or bbox_idx >= len(self.bboxes):
            return self.HANDLE_NONE
        
        bbox = self.bboxes[bbox_idx]['bbox']
        x1, y1 = self._img_to_widget(bbox[0], bbox[1])
        x2, y2 = self._img_to_widget(bbox[2], bbox[3])
        
        hs = self.handle_size
        handles = [
            (x1, y1, self.HANDLE_TL),
            (x2, y1, self.HANDLE_TR),
            (x2, y2, self.HANDLE_BR),
            (x1, y2, self.HANDLE_BL),
        ]
        
        for hx, hy, handle_id in handles:
            if abs(wx - hx) <= hs and abs(wy - hy) <= hs:
                return handle_id
        
        # 检查是否在BBox内部（用于移动）
        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return self.HANDLE_CENTER
        
        return self.HANDLE_NONE
    
    def _get_bbox_at(self, wx: int, wy: int) -> int:
        """获取窗口坐标处的BBox索引"""
        for i, bbox_dict in enumerate(self.bboxes):
            bbox = bbox_dict['bbox']
            x1, y1 = self._img_to_widget(bbox[0], bbox[1])
            x2, y2 = self._img_to_widget(bbox[2], bbox[3])
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                return i
        return -1
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 背景
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        # 绘制图像
        if self.pixmap:
            self._update_transform()
            scaled_pixmap = self.pixmap.scaled(
                int(self.img_w * self.scale),
                int(self.img_h * self.scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), scaled_pixmap)
        
        # 绘制所有BBox
        for i, bbox_dict in enumerate(self.bboxes):
            self._draw_bbox(painter, bbox_dict, i == self.selected_idx)
        
        # 绘制正在绘制的BBox
        if self.drawing and self.draw_start and self.draw_end:
            temp_bbox = {
                'bbox': [
                    min(self.draw_start[0], self.draw_end[0]),
                    min(self.draw_start[1], self.draw_end[1]),
                    max(self.draw_start[0], self.draw_end[0]),
                    max(self.draw_start[1], self.draw_end[1]),
                ],
                'class_id': self.current_class_id,
                'confidence': 1.0
            }
            self._draw_bbox(painter, temp_bbox, True, is_temp=True)
        
        painter.end()
    
    def _draw_bbox(self, painter: QPainter, bbox_dict: Dict, selected: bool, is_temp: bool = False):
        """绘制单个BBox"""
        bbox = bbox_dict['bbox']
        class_id = bbox_dict.get('class_id', 0)
        confidence = bbox_dict.get('confidence', 1.0)
        
        # 获取颜色
        color = COLORS[class_id % len(COLORS)]
        qcolor = QColor(color[0], color[1], color[2])
        
        # 转换坐标
        x1, y1 = self._img_to_widget(bbox[0], bbox[1])
        x2, y2 = self._img_to_widget(bbox[2], bbox[3])
        
        # 绘制矩形
        pen = QPen(qcolor, 3 if selected else 2)
        if is_temp:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        
        # 选中时绘制调整手柄
        if selected and not is_temp:
            hs = self.handle_size
            brush = QBrush(qcolor)
            painter.setBrush(brush)
            for hx, hy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                painter.drawRect(hx - hs//2, hy - hs//2, hs, hs)
            painter.setBrush(Qt.NoBrush)
        
        # 绘制标签
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
        label = f"{class_name} {confidence:.2f}"
        
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        # 标签背景
        text_rect = painter.fontMetrics().boundingRect(label)
        bg_rect = QRect(x1, y1 - text_rect.height() - 4, text_rect.width() + 8, text_rect.height() + 4)
        painter.fillRect(bg_rect, qcolor)
        
        # 标签文字
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x1 + 4, y1 - 4, label)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() != Qt.LeftButton or self.image is None:
            return
        
        wx, wy = event.x(), event.y()
        ix, iy = self._widget_to_img(wx, wy)
        
        # 检查是否点击在已有BBox上
        clicked_idx = self._get_bbox_at(wx, wy)
        
        if self.selected_idx >= 0:
            # 已选中状态，检查是否点击调整手柄
            handle = self._get_handle_at(wx, wy, self.selected_idx)
            if handle != self.HANDLE_NONE:
                if handle == self.HANDLE_CENTER:
                    # 开始移动
                    self.moving = True
                    bbox = self.bboxes[self.selected_idx]['bbox']
                    self.drag_offset = (ix - bbox[0], iy - bbox[1])
                else:
                    # 开始调整大小
                    self.resizing = True
                    self.active_handle = handle
                return
        
        if clicked_idx >= 0:
            # 点击了一个BBox，选中它
            self.selected_idx = clicked_idx
            self.bbox_selected.emit(clicked_idx)
            self.update()
        elif self.mode == self.MODE_DRAW:
            if self.click_mode:
                # 点击放置模式：直接在点击位置创建固定大小的框
                half_w = self.fixed_width // 2
                half_h = self.fixed_height // 2
                x1 = max(0, ix - half_w)
                y1 = max(0, iy - half_h)
                x2 = min(self.img_w, ix + half_w)
                y2 = min(self.img_h, iy + half_h)
                
                new_bbox = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': self.current_class_id,
                    'confidence': 1.0
                }
                self.bboxes.append(new_bbox)
                self.selected_idx = len(self.bboxes) - 1
                self.bbox_created.emit(new_bbox)
                self.annotations_changed.emit()
                self.update()
            else:
                # 拖动绘制模式：开始绘制新BBox
                self.drawing = True
                self.draw_start = (ix, iy)
                self.draw_end = (ix, iy)
                self.selected_idx = -1
        else:
            # 取消选择
            self.selected_idx = -1
            self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.image is None:
            return
        
        wx, wy = event.x(), event.y()
        ix, iy = self._widget_to_img(wx, wy)
        
        if self.drawing:
            # 更新绘制终点
            self.draw_end = (ix, iy)
            self.update()
        elif self.moving and self.selected_idx >= 0:
            # 移动BBox
            bbox = self.bboxes[self.selected_idx]['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            new_x1 = ix - self.drag_offset[0]
            new_y1 = iy - self.drag_offset[1]
            # 边界限制
            new_x1 = max(0, min(self.img_w - w, new_x1))
            new_y1 = max(0, min(self.img_h - h, new_y1))
            self.bboxes[self.selected_idx]['bbox'] = [new_x1, new_y1, new_x1 + w, new_y1 + h]
            self.update()
        elif self.resizing and self.selected_idx >= 0:
            # 调整BBox大小
            bbox = self.bboxes[self.selected_idx]['bbox']
            if self.active_handle == self.HANDLE_TL:
                bbox[0], bbox[1] = ix, iy
            elif self.active_handle == self.HANDLE_TR:
                bbox[2], bbox[1] = ix, iy
            elif self.active_handle == self.HANDLE_BR:
                bbox[2], bbox[3] = ix, iy
            elif self.active_handle == self.HANDLE_BL:
                bbox[0], bbox[3] = ix, iy
            # 确保坐标有效
            if bbox[0] > bbox[2]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[3], bbox[1]
            self.update()
        else:
            # 更新鼠标光标
            if self.selected_idx >= 0:
                handle = self._get_handle_at(wx, wy, self.selected_idx)
                if handle in (self.HANDLE_TL, self.HANDLE_BR):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif handle in (self.HANDLE_TR, self.HANDLE_BL):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif handle == self.HANDLE_CENTER:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.CrossCursor if self.mode == self.MODE_DRAW else Qt.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() != Qt.LeftButton:
            return
        
        if self.drawing and self.draw_start and self.draw_end:
            # 完成绘制
            x1 = min(self.draw_start[0], self.draw_end[0])
            y1 = min(self.draw_start[1], self.draw_end[1])
            x2 = max(self.draw_start[0], self.draw_end[0])
            y2 = max(self.draw_start[1], self.draw_end[1])
            
            # 最小尺寸检查
            if x2 - x1 > 5 and y2 - y1 > 5:
                new_bbox = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': self.current_class_id,
                    'confidence': 1.0
                }
                self.bboxes.append(new_bbox)
                self.selected_idx = len(self.bboxes) - 1
                self.bbox_created.emit(new_bbox)
                self.annotations_changed.emit()
            
            self.drawing = False
            self.draw_start = None
            self.draw_end = None
            self.update()
        
        if self.moving or self.resizing:
            self.bbox_modified.emit(self.selected_idx)
            self.annotations_changed.emit()
        
        self.moving = False
        self.resizing = False
        self.active_handle = self.HANDLE_NONE
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected()
        elif event.key() == Qt.Key_Escape:
            self.selected_idx = -1
            self.drawing = False
            self.update()
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        self._update_transform()
        self.update()
    
    def contextMenuEvent(self, event):
        """右键菜单"""
        if self.selected_idx < 0:
            return
        
        menu = QMenu(self)
        
        # 删除选项
        delete_action = QAction("删除", self)
        delete_action.triggered.connect(self.delete_selected)
        menu.addAction(delete_action)
        
        # 更改类别子菜单
        if self.class_names:
            class_menu = menu.addMenu("更改类别")
            for i, name in enumerate(self.class_names):
                action = QAction(name, self)
                action.triggered.connect(lambda checked, cid=i: self.change_selected_class(cid))
                class_menu.addAction(action)
        
        menu.exec_(event.globalPos())
