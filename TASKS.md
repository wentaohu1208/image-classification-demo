# Image Classification Demo - Task List

## 已完成 (Completed)

- [x] 项目基础架构搭建
  - [x] 模块化目录结构 (src/{data,model,trainer}_module)
  - [x] Hydra 配置系统
  - [x] Factory & Registry 模式

- [x] 核心功能实现
  - [x] ResNet-18 模型 (适配 CIFAR-10 32x32)
  - [x] CIFAR-10 数据模块
  - [x] 训练循环 (支持 checkpoint, early stopping)
  - [x] 评估脚本

- [x] 配置与文档
  - [x] Hydra YAML 配置 (data/model/trainer)
  - [x] README.md 完整文档
  - [x] pyproject.toml 包配置

- [x] 工程化
  - [x] Git 仓库初始化
  - [x] GitHub 远程仓库创建与推送

## 进行中 (In Progress)

- [ ] 项目管理文件创建
  - [x] .claude/CLAUDE.md 项目指南
  - [x] TASKS.md 任务列表
  - [ ] .gitignore 完善

## 待办 (Todo)

### 功能增强

- [ ] 添加更多模型
  - [ ] ResNet-34, ResNet-50
  - [ ] VGG 系列
  - [ ] MobileNet (轻量级)

- [ ] 数据增强
  - [ ] AutoAugment
  - [ ] Cutout/RandomErasing
  - [ ] Mixup/CutMix

- [ ] 训练优化
  - [ ] 学习率调度器 (Cosine, Step, Plateau)
  - [ ] 优化器选择 (Adam, AdamW, SGD)
  - [ ] 标签平滑 (Label Smoothing)

### 实验管理

- [ ] Weights & Biases 集成
- [ ] TensorBoard 日志
- [ ] 超参数搜索 (Optuna)

### 测试与CI

- [ ] 单元测试 (pytest)
  - [ ] 数据集测试
  - [ ] 模型前向传播测试
  - [ ] 训练流程测试

- [ ] GitHub Actions
  - [ ] 代码格式检查 (black, ruff)
  - [ ] 类型检查 (mypy)
  - [ ] 测试运行

### 文档

- [ ] API 文档 (docstring)
- [ ] 教程笔记本 (Jupyter)
- [ ] 训练报告模板

### 部署

- [ ] ONNX 导出
- [ ] TensorRT 优化
- [ ] 推理服务示例

## 计划与优先级

### Phase 1: 完善基础 (当前)
- 完善工程化配置 (.gitignore, pre-commit)
- 添加基础测试

### Phase 2: 功能扩展
- 更多模型架构
- 高级数据增强
- 实验追踪

### Phase 3: 生产就绪
- 模型导出与优化
- 推理服务
- 完整文档

## 备注

- 预期 CIFAR-10 准确率: 88-92% (ResNet-18)
- 训练时间: ~10-15分钟 (单 GPU)
- 环境: conda + PyTorch CUDA
