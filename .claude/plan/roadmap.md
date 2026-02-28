# Project Roadmap

## Phase 1: Foundation (当前)
**目标**: 完善项目基础工程化配置

### Week 1
- [x] 创建项目管理文件 (CLAUDE.md, TASKS.md)
- [x] 添加 .gitignore
- [ ] 添加 pre-commit hooks
- [ ] 配置代码格式化工具 (black, ruff)
- [ ] 编写基础单元测试

**里程碑**: 代码质量检查自动化

## Phase 2: Features
**目标**: 扩展模型和训练功能

### Week 2-3
- [ ] 添加 ResNet-34/50
- [ ] 实现数据增强 (AutoAugment, Cutout)
- [ ] 集成 Weights & Biases
- [ ] 学习率调度器优化

**里程碑**: 支持多种模型架构，训练过程可追踪

## Phase 3: Production
**目标**: 模型导出与部署准备

### Week 4
- [ ] ONNX 导出脚本
- [ ] TensorRT 优化
- [ ] 推理 API 示例
- [ ] 完整文档与教程

**里程碑**: 模型可部署到生产环境

## 长期规划

- 支持更多数据集 (ImageNet, CIFAR-100)
- 分布式训练支持
- 模型压缩与量化
- 自动化超参数搜索
