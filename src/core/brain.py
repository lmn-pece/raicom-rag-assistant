import re
import torch
import threading
from transformers import TextIteratorStreamer
from src.config import AGENT_SYSTEM_PROMPT
from src.core.agent_logic import TOOLS


class RAICOMAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        self.max_turns = 5

    def think_and_act(self, user_input):
        self.history.append({"role": "user", "content": user_input})

        for turn in range(self.max_turns):
            print(f"\n[Turn {turn + 1}] 🤖 思考中: ", end="", flush=True)

            # 1. 准备流式输出器
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 2. 构造输入
            text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # 3. 在后台线程运行模型生成
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                stop_strings=["Observation:"],
                tokenizer=self.tokenizer
            )
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 4. 主线程实时从 streamer 中获取 token 并打印
            full_response = ""
            for new_text in streamer:
                # 检查是否生成了停止词（有时 streamer 会带出一部分停止词）
                if "Observation:" in new_text:
                    break
                print(new_text, end="", flush=True)
                full_response += new_text

            # 等待线程结束
            thread.join()

            # 5. 解析 Action (逻辑同前)
            action_match = re.search(r"Action:\s*(\w+)", full_response)
            action_input_match = re.search(r"Action Input:\s*(.*)", full_response)

            if action_match and action_input_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_input_match.group(1).strip()

                if tool_name in TOOLS:
                    print(f"\n🛠️  执行工具: {tool_name}")
                    observation = TOOLS[tool_name](tool_input)
                    print(f"👁️  观察结果: {observation}")

                    self.history.append({"role": "assistant", "content": full_response})
                    self.history.append({"role": "user", "content": f"Observation: {observation}"})
                    continue

            # Final Answer 落地
            self.history.append({"role": "assistant", "content": full_response})
            return full_response
        return None