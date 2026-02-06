# -*- coding: utf-8 -*-
"""
YOLOv5 半自动图像标注工具
应用入口
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from main_window import MainWindow


def main():
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，跨平台一致
    
    window = MainWindow()
    
    # 设置窗口透明度 (0.0 完全透明 - 1.0 完全不透明)
    window.setWindowOpacity(0.9)
    
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
