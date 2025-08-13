import json

def split_sharegpt_jsonl_sequential(
    input_path: str,
    train_path: str,
    test_path: str,
    train_count: int
):
    # 1. 按行读取 JSONL
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))

    # 2. 顺序切分
    train_data = data[:train_count]
    test_data = data[train_count:]

    # 3. 保存为 JSONL
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"总样本数: {len(data)}")
    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    print(f"已保存到:\n  {train_path}\n  {test_path}")



if __name__ == "__main__":
    input_file = "/home/qjsys/lly/SpecForge/data/sharegpt/processed/sharegpt.jsonl"          # 原文件路径
    output_train = "/home/qjsys/lly/SpecForge/data/sharegpt/processed/sharegpt_train.json"  # 训练集输出路径
    output_test = "/home/qjsys/lly/SpecForge/data/sharegpt/processed/sharegpt_eval.json"    # 测试集输出路径

    split_sharegpt_jsonl_sequential(
        input_file,
        output_train,
        output_test,
        train_count=100000  # 前100000条作为训练集
    )
