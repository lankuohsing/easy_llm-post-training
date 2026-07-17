from datasets import load_dataset
import os

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:18669'  # 将端口号替换成你的实际端口
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:18669' # 将端口号替换成你的实际端口

# datasets>=4 不再执行 Hub 上的数据集脚本（logiqa.py）；使用 Hub 自动生成的 Parquet 分支加载。
dataset = load_dataset(
    "lucasmccabe/logiqa",
    revision="refs/convert/parquet",
    # split="train",
    cache_dir="./.cache",
)
dataset.save_to_disk("/Users/guoxing.lan/projects/dataset/logiqa")
