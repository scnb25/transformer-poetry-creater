import os
import json
import re

def clean_text(s):
    # 只保留中文
    return re.sub(r"[^\u4e00-\u9fa5]", "", s)

def load_poetry(root_dir):
    all_sentences = []

    # 遍历所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)

                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        poems = json.load(f)

                        for poem in poems:
                            paragraphs = poem.get("paragraphs", [])

                            # 拼接一首诗
                            text = "".join(paragraphs)
                            text = clean_text(text)

                            if len(text) > 10:  # 过滤太短
                                all_sentences.append(text)

                except Exception as e:
                    print("读取失败:", path)

    return all_sentences

if __name__ == "__main__":
    data = load_poetry(r"D:\transformer\dataset\chinese-poetry-master")

    print("数据量:", len(data))
    print("示例:", data[0])