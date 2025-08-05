import time
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer


def build_prompt(question: str):
    conv = get_conversation_template(model_id)   # 拿到模板
    conv.append_message(conv.roles[0], question) # user 消息
    conv.append_message(conv.roles[1], None)     # 预留 assistant 槽
    prompt = conv.get_prompt()                   # 拼接完整 prompt
    return prompt, conv

if __name__ == "__main__":
    model_id = 'vicuna-7b'
    question = (
        "Janet’s ducks lay 16 eggs per day. She eats three and uses four for "
        "muffins. She sells the rest at $2 each. How much does she earn daily?"
    )

    prompt, conv = build_prompt(question)
    print(type(prompt))
    print(prompt)