# 问题修复报告

本目录记录 StoryWeaver 项目各类问题的修复过程和解决方案。

## 文件说明

### DistilBERT 相关修复
- **distilbert-compatibility-fix.md** - DistilBERT 兼容性修复完整总结，包含 5 层防护方案
- **distilbert-tokenizer-fix.md** - DistilBERT Tokenizer 与 Model 输入兼容性加固报告
- **distilbert-troubleshooting.md** - DistilBERT 问题排查指南与解决方案

### 其他修复
- **fastcoref-fix.md** - FastCoref 与 transformers 5.2.0 不兼容性修复方案

## 修复模式

项目中采用的通用修复模式：
1. **通用输入过滤** - 按模型 forward 签名自动过滤输入字段
2. **防御性配置** - 从源头减少无效字段生成
3. **安全包装** - 重试 + 回退 + 版本警告机制
4. **依赖锁定** - 确保测试范围内运行
5. **健康检查** - 提前发现环境问题