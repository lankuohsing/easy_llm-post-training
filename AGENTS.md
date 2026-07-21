# Repository Guidelines

## 项目结构与模块组织

本仓库是一个面向 LLM 后训练的教程项目。每个训练主题应放在独立的顶层目录中：

- `sft/`：监督微调相关笔记与示例。
- `dpo/`：Direct Preference Optimization 相关笔记与示例。
- `opd/`：在线偏好学习、蒸馏等相关材料。
- `rl/`：强化学习与 RLHF 风格训练材料。
- `utils/`：通用辅助脚本，目前包含 `utils/dataset_utils/`。

目录和文件名优先使用小写。主题入口文档优先使用 `readme.md`，除非后续全仓库统一改为 `README.md`。

## 构建、测试与开发命令

当前没有全仓库级别的构建系统。小型工具脚本可直接用 Python 运行：

```bash
python utils/dataset_utils/download_data_from_hf.py
python utils/dataset_utils/load_dataset_from_disk.py
```

新增训练命令前，应在对应主题的 `readme.md` 中说明依赖、GPU/内存需求、输入路径和输出路径。

## 代码风格与命名规范

Python 代码应保持简单、清晰，适合教程阅读。使用 4 空格缩进、描述性的 `snake_case` 命名；脚本超过短示例规模后，应拆分为小函数。提交示例代码时避免写死个人绝对路径，优先使用变量、环境变量，或明确标注的占位路径，例如 `/path/to/dataset`。

后续文档、代码注释和教程说明默认使用中文。专业术语可保留常见英文写法，例如 SFT、DPO、RLHF、LoRA、Hugging Face。

不同章节中的公式符号必须兼容、一致。新增或改写公式时，应检查已有章节的变量命名、上下标和目标函数记号，避免同一概念使用多个符号，或同一符号表示不同含义。

Markdown 公式应同时兼容 GitHub 与 VS Code 预览。行内公式使用 `$...$` 时，开头的 `$` 不要紧贴中文标点，应与前文保留一个空格；不要在 `$` 与公式内容之间额外加空格。行间公式的 `$$` 必须单独成行，并在公式块前后保留空行。LaTeX 公式中不要直接写 `<`，因为 GitHub 可能先将其解析为 HTML；小于号统一写成 `\lt`。仓库通过 `.vscode/settings.json` 启用 VS Code 内置 Markdown 数学渲染。修改公式后，应检查是否仍有紧贴中文标点的行内公式或数学环境中的裸 `<`。

## 测试规范

当前尚未配置正式测试框架。向 `utils/` 添加可复用代码时，应提供可本地运行的轻量检查或示例。若后续引入测试套件，请放在 `tests/` 目录下，测试文件命名为 `test_*.py`，并记录运行命令，例如：

```bash
pytest
```

教程命令优先提供小型冒烟测试，用于验证依赖导入、数据集加载和输出目录创建。

## 提交与 Pull Request 规范

当前 Git 历史只有初始提交，后续请使用清晰的约定式提交信息，例如 `docs: expand sft tutorial` 或 `feat: add dataset loader utility`。

Pull Request 应包含简短摘要、变更目录、已运行命令，以及刻意排除在 Git 之外的生成产物说明。如有关联 issue，请一并链接。不要提交模型权重、checkpoint、本地数据集、缓存、日志或私密凭据。

## Agent 专用说明

Codex 或其他 AI agent 在编辑前应先检查仓库结构，并保持改动范围聚焦。优先保证教程可读性，不要为了“聪明”的抽象牺牲清晰度。若同时修改代码和文档，必须确保文档中的命令与实际脚本保持一致。
