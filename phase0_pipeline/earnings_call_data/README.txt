# Earnings Call 数据集说明

## 数据来源

数据集地址：
https://huggingface.co/datasets/RudrakshNanavaty/earnings-call-data

## 下载方式

```bash
# 安装 huggingface datasets
pip install datasets

# 下载并导出为 Parquet
python -c "
from datasets import load_dataset
ds = load_dataset('RudrakshNanavaty/earnings-call-data')
ds['train'].to_parquet('earnings_call_data/earnings-call-data.parquet')
"
```

或直接从 HuggingFace 网页下载 Parquet 文件放入本目录。

## 数据格式要求

目录中需要包含至少一个 `.parquet` 文件，且包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `symbol` | string | 股票代码 |
| `earnings_date` | date/string | 财报日期 |
| `earnings_transcript` | string | 完整电话会议转录文本 |
| `press_release_ex991` | string（可选） | 8-K 新闻稿正文 |

## 预期规模

- 183,000 行（~685 只股票 × 多年季度）
- 文件大小：~500-800 MB（Parquet 压缩后）

下载后直接放入此目录，运行 `python build_transcript_index.py` 即可。
