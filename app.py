from src.core.brain import RAICOMAgent
from src.model_loader import load_qwen


def _print_sources(chunks) -> None:
    if not chunks:
        return
    print("\n\n本地资料依据:")
    for idx, c in enumerate(chunks, start=1):
        print(
            f"  [来源{idx}] {c.get('file_name')} 行 {c.get('line_start')}-{c.get('line_end')} "
            f"chunk_id={c.get('chunk_id')}"
        )


def main() -> None:
    model, tokenizer = load_qwen()
    agent = RAICOMAgent(model, tokenizer)

    print("\n本地 RAG 助手已就绪。输入 'exit' 退出，'clear' 清空对话。")
    while True:
        try:
            query = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "clear":
            agent.reset()
            print("对话记忆已清空。")
            continue
        if not query:
            continue

        print("\nAssistant: ", end="", flush=True)
        answer = agent.answer(query, stream=True)
        if not agent.last_streamed:
            print(answer)
        _print_sources(agent.last_chunks)
        print()


if __name__ == "__main__":
    main()
