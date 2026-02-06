# -*- coding: utf-8 -*-
"""
项目管理模块
处理项目创建、加载、保存和最近项目记录
"""
import os
import json
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# 配置文件路径
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".yolov5auto")
RECENT_PROJECTS_FILE = os.path.join(CONFIG_DIR, "recent_projects.json")
MAX_RECENT_PROJECTS = 10


class Project:
    """训练项目类"""
    
    # 项目目录结构
    ALL_IMAGES_DIR = "all_images"      # 所有待标注图片
    IMAGES_DIR = "images"              # 已标注的图片
    LABELS_DIR = "labels"              # 标注文件
    CLASSES_FILE = "classes.txt"       # 类别文件
    PROJECT_FILE = "project.json"      # 项目配置文件
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.name = os.path.basename(project_path)
        self.classes: List[str] = []
        self.created_at: str = ""
        self.model_path: str = ""
        
    @property
    def all_images_dir(self) -> str:
        return os.path.join(self.project_path, self.ALL_IMAGES_DIR)
    
    @property
    def images_dir(self) -> str:
        return os.path.join(self.project_path, self.IMAGES_DIR)
    
    @property
    def labels_dir(self) -> str:
        return os.path.join(self.project_path, self.LABELS_DIR)
    
    @property
    def classes_file(self) -> str:
        return os.path.join(self.project_path, self.CLASSES_FILE)
    
    @property
    def project_file(self) -> str:
        return os.path.join(self.project_path, self.PROJECT_FILE)
    
    def create(self, classes: List[str] = None):
        """创建新项目"""
        # 创建目录结构
        os.makedirs(self.all_images_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # 设置类别
        self.classes = classes or ["object"]
        self.created_at = datetime.now().isoformat()
        
        # 保存类别文件
        self._save_classes()
        
        # 保存项目配置
        self._save_config()
    
    def load(self) -> bool:
        """加载项目"""
        if not os.path.exists(self.project_file):
            # 尝试从旧格式迁移
            if os.path.exists(self.classes_file):
                self._load_classes()
                self.created_at = ""
                return True
            return False
        
        try:
            with open(self.project_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.classes = config.get('classes', ['object'])
            self.created_at = config.get('created_at', '')
            self.model_path = config.get('model_path', '')
            return True
        except Exception:
            return False
    
    def save(self):
        """保存项目配置"""
        self._save_classes()
        self._save_config()
    
    def _save_classes(self):
        """保存类别文件"""
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")
    
    def _load_classes(self):
        """加载类别文件"""
        if os.path.exists(self.classes_file):
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = ['object']
    
    def _save_config(self):
        """保存项目配置"""
        config = {
            'name': self.name,
            'classes': self.classes,
            'created_at': self.created_at,
            'model_path': self.model_path
        }
        with open(self.project_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def add_images(self, image_paths: List[str]) -> int:
        """
        添加图片到项目
        将图片复制到 all_images 目录
        返回成功添加的图片数量
        """
        added = 0
        for src_path in image_paths:
            if not os.path.isfile(src_path):
                continue
            
            filename = os.path.basename(src_path)
            dst_path = os.path.join(self.all_images_dir, filename)
            
            # 如果文件已存在，添加后缀
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                i = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(self.all_images_dir, f"{name}_{i}{ext}")
                    i += 1
            
            try:
                shutil.copy2(src_path, dst_path)
                added += 1
            except Exception:
                pass
        
        return added
    
    def get_all_images(self) -> List[str]:
        """获取所有待标注图片"""
        return self._get_images_in_dir(self.all_images_dir)
    
    def get_labeled_images(self) -> List[str]:
        """获取已标注图片"""
        return self._get_images_in_dir(self.images_dir)
    
    def _get_images_in_dir(self, directory: str) -> List[str]:
        """获取目录中的图片文件"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if not os.path.isdir(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                files.append(os.path.join(directory, filename))
        return sorted(files)
    
    def mark_as_labeled(self, image_path: str):
        """
        标记图片为已标注
        将图片从 all_images 复制到 images
        """
        if not os.path.exists(image_path):
            return
        
        filename = os.path.basename(image_path)
        src_dir = os.path.dirname(image_path)
        
        # 如果图片在 all_images 中，复制到 images
        if src_dir == self.all_images_dir:
            dst_path = os.path.join(self.images_dir, filename)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
    
    def get_unlabeled_count(self) -> int:
        """获取未标注图片数量"""
        all_count = len(self.get_all_images())
        labeled_count = len(self.get_labeled_images())
        return max(0, all_count - labeled_count)
    
    def is_image_labeled(self, image_path: str) -> bool:
        """检查图片是否已标注"""
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        label_path = os.path.join(self.labels_dir, f"{name}.txt")
        return os.path.exists(label_path) and os.path.getsize(label_path) > 0


class RecentProjectsManager:
    """最近项目管理器"""
    
    def __init__(self):
        self._ensure_config_dir()
        self.projects: List[Dict] = []
        self.load()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        os.makedirs(CONFIG_DIR, exist_ok=True)
    
    def load(self):
        """加载最近项目列表"""
        if os.path.exists(RECENT_PROJECTS_FILE):
            try:
                with open(RECENT_PROJECTS_FILE, 'r', encoding='utf-8') as f:
                    self.projects = json.load(f)
            except Exception:
                self.projects = []
        else:
            self.projects = []
    
    def save(self):
        """保存最近项目列表"""
        with open(RECENT_PROJECTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.projects, f, ensure_ascii=False, indent=2)
    
    def add_project(self, project_path: str, name: str = None):
        """添加项目到最近列表"""
        project_path = os.path.abspath(project_path)
        name = name or os.path.basename(project_path)
        
        # 移除已存在的相同路径
        self.projects = [p for p in self.projects if p.get('path') != project_path]
        
        # 添加到列表开头
        self.projects.insert(0, {
            'path': project_path,
            'name': name,
            'last_opened': datetime.now().isoformat()
        })
        
        # 限制数量
        self.projects = self.projects[:MAX_RECENT_PROJECTS]
        
        self.save()
    
    def get_recent_projects(self) -> List[Dict]:
        """获取最近项目列表"""
        # 过滤掉不存在的项目
        valid_projects = []
        for p in self.projects:
            if os.path.exists(p.get('path', '')):
                valid_projects.append(p)
        
        if len(valid_projects) != len(self.projects):
            self.projects = valid_projects
            self.save()
        
        return self.projects
    
    def clear(self):
        """清空最近项目列表"""
        self.projects = []
        self.save()
