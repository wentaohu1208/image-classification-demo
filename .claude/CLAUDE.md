# Image Classification Demo - Project Guide

## Project Overview

基于PyTorch和ResNet的CIFAR-10图像分类项目，采用模块化架构和Hydra配置管理。

## Quick Reference

### 常用命令

```bash
# 激活环境
conda activate imgcls

# 训练
python train.py

# 评估
python evaluate.py --checkpoint outputs/.../best_model.pt

# 使用自定义配置
python train.py trainer.epochs=100 trainer.lr=0.01
```

### 项目结构

```
image_classification_demo/
├── conf/                      # Hydra配置
├── src/
│   ├── data_module/           # 数据加载
│   ├── model_module/          # 模型定义
│   ├── trainer_module/        # 训练循环
│   └── utils/                 # 工具函数
├── train.py                   # 训练入口
└── evaluate.py                # 评估入口
```

## Development Guidelines

### 添加新数据集

1. 在 `src/data_module/dataset.py` 添加数据集构建器
2. 创建配置 `conf/data/<name>.yaml`
3. 运行: `python train.py data=<name>`

### 添加新模型

1. 在 `src/model_module/` 创建模型类
2. 创建配置 `conf/model/<name>.yaml`
3. 运行: `python train.py model=<name>`

### Code Style

- 文件行数限制: 200-400行
- 使用类型注解
- 使用 `__all__` 定义公开API
- 配置驱动，避免硬编码

## Git Workflow

- Commit 使用 Conventional Commits 规范
- Feature 分支: `feature/description`
- PR 合并到 master

## Active Tasks

See [TASKS.md](../TASKS.md) for current tasks and progress.
