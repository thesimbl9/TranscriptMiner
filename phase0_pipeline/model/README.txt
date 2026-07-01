# BGE-M3 模型下载说明

本 Pipeline 使用 BAAI/bge-m3 作为文本向量化模型（1024 维）。

## 方式一：联网自动下载（推荐）

删除或留空 model/ 目录，脚本运行时会自动从 HuggingFace 下载模型到缓存。
首次下载约需 2.2GB，之后缓存复用。

不需要任何额外操作，直接运行脚本即可。

## 方式二：离线本地加载

如果需要在无网络环境运行，预先下载模型到 model/ 目录：

```bash
# 安装 git-lfs
git lfs install

# 克隆模型到 model/ 目录
git clone https://huggingface.co/BAAI/bge-m3 model/
```

下载后 model/ 目录应包含：
```
model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── sentence_bert_config.json
├── model.safetensors          # ~2.2GB
└── ...（其他配置文件）
```

脚本会自动检测 model/ 中是否有模型文件，有则离线加载，无则联网下载。
