# -*- coding: utf-8 -*-
"""
孪生分类标注界面 - 包含多种工作模式
模式1/3: 分离式（bg.jpg + icon*.png）
模式2: 单图式（一张图包含上方目标+下方提示）
模式4: 提示词式（仅背景图，从文件名解析类别）
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

# 导入各模式的组件
from siamese_folder_widget import SiameseFolderWidget
from single_image_widget import SingleImageWidget
from hint_mode_widget import HintModeWidget


class SiameseWidget(QWidget):
    """孪生分类标注主组件 - Tab 切换多种模式"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 子 Tab
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)
        
        # 模式1/3: 分离式
        self.folder_widget = SiameseFolderWidget()
        self.sub_tabs.addTab(self.folder_widget, "模式1/3: 分离式")
        
        # 模式2: 单图式
        self.single_widget = SingleImageWidget()
        self.sub_tabs.addTab(self.single_widget, "模式2: 单图式")
        
        # 模式4: 提示词式
        self.hint_widget = HintModeWidget()
        self.sub_tabs.addTab(self.hint_widget, "模式4: 提示词式")

