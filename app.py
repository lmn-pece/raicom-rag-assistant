# app.py
from src.model_loader import load_qwen
from src.core.brain import RAICOMAgent


def main():
    # 1. 硬件/模型初始化
    model, tokenizer = load_qwen()

    # 2. 实例化大脑
    agent = RAICOMAgent(model, tokenizer)

    print("\n✅ RAICOM 全栈智能体重构完成。")
    print("输入 'exit' 退出，输入 'clear' 重置记忆。")

    while True:
        query = input("\n👤 指挥官指令: ").strip()

        if query.lower() in ['exit', 'quit']:
            break
        if query.lower() == 'clear':
            agent.history = agent.history[:1]  # 只保留 System Prompt
            print("✨ 记忆已清空")
            continue
        if not query:
            continue

        # 3. 让大脑处理逻辑
        final_response = agent.think_and_act(query)

        print("\n" + "-"*30)


if __name__ == "__main__":
    main()