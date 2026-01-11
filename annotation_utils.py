# -*- coding: utf-8 -*-
"""
标注文件工具函数
支持YOLO格式 .txt 文件的读写和坐标转换
"""
import os
from typing import List, Dict, Tuple, Optional


def load_classes(classes_file: str) -> List[str]:
    """
    加载类别名称列表
    
    Args:
        classes_file: classes.txt 文件路径
        
    Returns:
        类别名称列表
    """
    if not os.path.exists(classes_file):
        return []
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_classes(classes_file: str, class_names: List[str]):
    """
    保存类别名称列表
    
    Args:
        classes_file: classes.txt 文件路径
        class_names: 类别名称列表
    """
    with open(classes_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")


def bbox_to_yolo(bbox: List[int], img_w: int, img_h: int, class_id: int) -> str:
    """
    将 [x1, y1, x2, y2] 转换为 YOLO格式字符串
    
    Args:
        bbox: 边界框 [x1, y1, x2, y2] (像素坐标)
        img_w: 图像宽度
        img_h: 图像高度
        class_id: 类别ID
        
    Returns:
        YOLO格式字符串: "class_id center_x center_y width height"
    """
    x1, y1, x2, y2 = bbox
    
    # 计算中心点和宽高
    center_x = (x1 + x2) / 2.0 / img_w
    center_y = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    
    # 裁剪到 [0, 1] 范围
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def yolo_to_bbox(yolo_line: str, img_w: int, img_h: int) -> Optional[Dict]:
    """
    将 YOLO格式行解析为 bbox 字典
    
    Args:
        yolo_line: YOLO格式字符串
        img_w: 图像宽度
        img_h: 图像高度
        
    Returns:
        {'class_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float}
        解析失败返回 None
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    
    try:
        class_id = int(parts[0])
        center_x = float(parts[1]) * img_w
        center_y = float(parts[2]) * img_h
        width = float(parts[3]) * img_w
        height = float(parts[4]) * img_h
        
        # 计算边界框
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # 裁剪到图像边界
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))
        
        return {
            'class_id': class_id,
            'bbox': [x1, y1, x2, y2],
            'confidence': 1.0  # 手动标注的置信度设为1
        }
    except (ValueError, IndexError):
        return None


def save_annotations(filepath: str, bboxes: List[Dict], img_w: int, img_h: int):
    """
    保存标注到 .txt 文件
    
    Args:
        filepath: 标注文件路径
        bboxes: 边界框列表 [{'class_id': int, 'bbox': [x1,y1,x2,y2], ...}]
        img_w: 图像宽度
        img_h: 图像高度
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for bbox_dict in bboxes:
            class_id = bbox_dict.get('class_id', 0)
            bbox = bbox_dict.get('bbox', [0, 0, 0, 0])
            line = bbox_to_yolo(bbox, img_w, img_h, class_id)
            f.write(line + '\n')


def load_annotations(filepath: str, img_w: int, img_h: int) -> List[Dict]:
    """
    从 .txt 文件加载标注
    
    Args:
        filepath: 标注文件路径
        img_w: 图像宽度
        img_h: 图像高度
        
    Returns:
        边界框列表
    """
    if not os.path.exists(filepath):
        return []
    
    bboxes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            bbox = yolo_to_bbox(line, img_w, img_h)
            if bbox:
                bboxes.append(bbox)
    
    return bboxes


def get_label_path(image_path: str, labels_dir: str = None) -> str:
    """
    获取图像对应的标注文件路径
    
    Args:
        image_path: 图像文件路径
        labels_dir: 标注目录（默认为图像同目录下的 labels/）
        
    Returns:
        标注文件路径
    """
    img_dir = os.path.dirname(image_path)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if labels_dir is None:
        labels_dir = os.path.join(img_dir, 'labels')
    
    return os.path.join(labels_dir, f"{img_name}.txt")


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> List[str]:
    """
    获取目录中的所有图像文件
    
    Args:
        directory: 目录路径
        extensions: 支持的图像扩展名
        
    Returns:
        图像文件路径列表（排序后）
    """
    if not os.path.isdir(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            files.append(os.path.join(directory, filename))
    
    return sorted(files)


def is_image_annotated(image_path: str, labels_dir: str = None) -> bool:
    """
    检查图像是否已标注
    
    Args:
        image_path: 图像文件路径
        labels_dir: 标注目录
        
    Returns:
        是否已标注
    """
    label_path = get_label_path(image_path, labels_dir)
    return os.path.exists(label_path) and os.path.getsize(label_path) > 0


if __name__ == '__main__':
    # 简单测试
    print("=== 坐标转换测试 ===")
    
    # 测试 bbox -> yolo
    bbox = [100, 50, 200, 150]
    img_w, img_h = 640, 480
    yolo_str = bbox_to_yolo(bbox, img_w, img_h, class_id=0)
    print(f"bbox {bbox} -> YOLO: {yolo_str}")
    
    # 测试 yolo -> bbox
    parsed = yolo_to_bbox(yolo_str, img_w, img_h)
    print(f"YOLO -> bbox: {parsed}")
    
    print("\n测试通过！")
