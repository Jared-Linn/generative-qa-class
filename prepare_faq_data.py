"""
从主项目 FAQ 数据生成 ChatML 格式训练数据

用法:  python prepare_faq_data.py
输出:  ./data/multi_turn_qa.json（覆盖原有 5 条样本）

FAQ 数据来源: ../智能问答系统/data/raw/faq_expanded.json（264 条）
"""

import json
import os
import sys

# 输入：主项目的 FAQ 数据
FAQ_FILE = "../智能问答系统/data/raw/faq_expanded.json"
# 也可以只用 25 条核心 FAQ
# FAQ_FILE = "../智能问答系统/data/raw/faq.json"

# 输出：训练数据
OUTPUT_FILE = "./data/multi_turn_qa.json"

SYSTEM_PROMPT = "你是一个专业的AI助手，擅长回答人工智能、机器学习、深度学习等技术问题。请用清晰、准确的语言回答问题。"


def main():
    # 检查 FAQ 文件是否存在
    if not os.path.exists(FAQ_FILE):
        print(f"❌ 找不到 FAQ 文件: {FAQ_FILE}")
        print("请确认路径是否正确，或修改 FAQ_FILE 变量")
        return

    # 加载 FAQ 数据
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    print(f"📂 已加载 {len(faq_data)} 条 FAQ")

    # 转换为 ChatML 格式
    formatted = []
    for item in faq_data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue

        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        formatted.append({"text": text})

    print(f"🔄 转换完成: {len(formatted)} 条")

    # 写入 JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 已保存到 {OUTPUT_FILE}")

    # 统计
    total_chars = sum(len(item["text"]) for item in formatted)
    avg_len = total_chars / len(formatted) if formatted else 0
    print(f"📊 总字符数: {total_chars}, 平均每条: {avg_len:.0f} 字")

    # 展示一条示例
    print("\n📝 示例:")
    print(formatted[0]["text"][:200] + "...")


if __name__ == "__main__":
    main()
