# -*- coding: utf-8 -*-
"""
YOLOv5 ONNX 目标检测器
基于 icon_click_solver.py 的实现，纯numpy，不依赖torch
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Union

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class YOLODetector:
    """YOLOv5 ONNX目标检测器 - 纯numpy实现"""
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, 
                 iou_thres: float = 0.45, class_names: Optional[List[str]] = None):
        """
        初始化检测器
        
        Args:
            model_path: ONNX模型文件路径
            conf_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
            class_names: 类别名称列表
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime 未安装或加载失败，请运行: pip install onnxruntime")
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = class_names or []
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        # 获取模型输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.img_size = input_shape[2] if len(input_shape) == 4 else 640
        
        # 预处理参数（在detect时更新）
        self.scale = 1.0
        self.pad_h = 0
        self.pad_w = 0
        self.orig_shape = (0, 0)
    
    def detect(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """
        对图像进行目标检测
        
        Args:
            image: 图像路径或numpy数组(BGR格式)
            
        Returns:
            检测结果列表 [{'class_id': int, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]
        """
        # 加载图像
        if isinstance(image, str):
            # 使用np.fromfile支持中文路径
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image.copy()
        
        self.orig_shape = img.shape[:2]
        
        # 预处理
        input_tensor = self._preprocess(img)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        
        # 后处理
        return self._postprocess_v5(outputs)
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        图像预处理：letterbox缩放 + 归一化
        
        Args:
            img: 输入图像(BGR格式)
            
        Returns:
            预处理后的张量 (1, 3, H, W)
        """
        h, w = img.shape[:2]
        
        # 计算缩放比例（保持宽高比）
        self.scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        
        # 缩放图像
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 创建画布并居中放置（letterbox）
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        self.pad_h = (self.img_size - new_h) // 2
        self.pad_w = (self.img_size - new_w) // 2
        canvas[self.pad_h:self.pad_h+new_h, self.pad_w:self.pad_w+new_w] = img_resized
        
        # BGR -> RGB, HWC -> CHW, 归一化
        canvas = canvas[:, :, ::-1].transpose(2, 0, 1)
        canvas = np.ascontiguousarray(canvas, dtype=np.float32) / 255.0
        
        return np.expand_dims(canvas, axis=0)
    
    def _postprocess_v5(self, outputs: np.ndarray) -> List[Dict]:
        """
        YOLOv5 后处理：置信度过滤 + NMS + 坐标转换
        
        Args:
            outputs: 模型输出
            
        Returns:
            检测结果列表
        """
        predictions = outputs[0]
        
        # 置信度过滤
        obj_conf = predictions[:, 4]
        mask = obj_conf > self.conf_thres
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        # 获取类别ID和置信度
        if predictions.shape[1] > 6:
            # 多类别模型：class_score = obj_conf * class_prob
            class_scores = predictions[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = predictions[:, 4] * class_scores[np.arange(len(class_scores)), class_ids]
        else:
            # 单类别模型
            confidences = predictions[:, 4]
            class_ids = predictions[:, 5].astype(int) if predictions.shape[1] > 5 else np.zeros(len(predictions), dtype=int)
        
        # xywh -> xyxy
        boxes = predictions[:, :4]
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 还原到原始图像坐标
        xyxy[:, [0, 2]] -= self.pad_w
        xyxy[:, [1, 3]] -= self.pad_h
        xyxy /= self.scale
        
        # 裁剪到图像边界
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, self.orig_shape[1])
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, self.orig_shape[0])
        
        # NMS
        indices = self._nms(xyxy, confidences)
        
        # 构建结果
        results = []
        for i in indices:
            result = {
                'class_id': int(class_ids[i]),
                'confidence': float(confidences[i]),
                'bbox': xyxy[i].astype(int).tolist()
            }
            # 添加类别名称（如果有）
            if self.class_names and result['class_id'] < len(self.class_names):
                result['class_name'] = self.class_names[result['class_id']]
            results.append(result)
        
        return results
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        非极大值抑制
        
        Args:
            boxes: 边界框 (N, 4) [x1, y1, x2, y2]
            scores: 置信度 (N,)
            
        Returns:
            保留的索引列表
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0)
            
            # 保留IoU小于阈值的框
            order = order[np.where(iou <= self.iou_thres)[0] + 1]
        
        return keep
    
    def set_thresholds(self, conf_thres: float = None, iou_thres: float = None):
        """更新阈值"""
        if conf_thres is not None:
            self.conf_thres = conf_thres
        if iou_thres is not None:
            self.iou_thres = iou_thres
    
    def set_class_names(self, class_names: List[str]):
        """设置类别名称"""
        self.class_names = class_names


if __name__ == '__main__':
    # 简单测试
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python yolo_detector.py <model.onnx> <image.jpg>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    detector = YOLODetector(model_path)
    results = detector.detect(image_path)
    
    print(f"检测到 {len(results)} 个目标:")
    for r in results:
        print(f"  类别{r['class_id']}: {r['bbox']}, 置信度: {r['confidence']:.2f}")
