import anthropic
import argparse
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from medusa.model.medusa_model import MedusaModel
import torch
from fastchat.model import get_conversation_template

parser = argparse.ArgumentParser()
parser.add_argument('--prompt_file', type=str, default='lib_prompt/prompt_original.txt')
parser.add_argument('--output_file', type=str, default='outputs/medusa_instant_gsm8k_test_greedy-2.txt', help='Output file for claude-instant')


model_id = 'vicuna-7b'
def build_prompt(question: str):
    conv = get_conversation_template(model_id)   # 拿到模板
    conv.append_message(conv.roles[0], question) # user 消息
    conv.append_message(conv.roles[1], None)     # 预留 assistant 槽
    prompt = conv.get_prompt()                   # 拼接完整 prompt
    return prompt, conv

gsm8k = load_dataset('gsm8k', 'main')
validation_index = np.load('lib_prompt/validation_index.npy')
validation_data = gsm8k['train'].select(validation_index)
gsm8k_test = gsm8k['test']

def parse_answer_file(answer_file):
    lines = open(answer_file, 'r').readlines()

    accuracy = 0
    last_number = 0
    should_find_answer = True

    for i, l in enumerate(lines):
        try:
            if should_find_answer:
                last_number = re.findall(r'\d+', l)[-1]
        except:
            pass

        if l.startswith('####'):
            reference_answer = l.split('####')[1].strip()
            if reference_answer == last_number:
                accuracy += 1
        elif l.startswith('===== CASE'):
            should_find_answer = True
        elif l.startswith('Reference Answer'):
            should_find_answer = False

    print('Accuracy: ', accuracy / len(gsm8k_test['question']) * 100)

def main(args):
    prompts = open(args.prompt_file, 'r').read()
    
    model = MedusaModel.from_pretrained(
        '/home/neil/sd/Medusa/ckpt/medusa-7b',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()



    run_count = 0
    with open(args.output_file, 'w') as f:
        for q, a in tqdm(zip(gsm8k_test['question'], gsm8k_test['answer']), total=len(gsm8k_test['question'])):
            # prompt = prompt + '\nQuestion: ' + q + '\n'

            # in_prompt =  prompts + '\nQuestion: ' + q + '\n' 
            # in_prompt = anthropic.HUMAN_PROMPT + "\n" + prompts + '\nQuestion: ' + q + '\n' + anthropic.AI_PROMPT
            in_prompt, _ =   build_prompt(q)
            run_count += 1
            input_ids = tokenizer.encode(in_prompt, return_tensors="pt").to(
                model.base_model.device
            )
            
            gen = model.medusa_generate(
                input_ids,
                temperature=0,
                max_steps=200,
                sampling='typical',
                fast=False,
            )

            # 2. 通过 for-in 迭代，不断从 generator 拉取每一段增量文本
            out_ans = ""
            prev_len = 0
            for chunk in gen:
                full = chunk["text"]
                # 取出上一次之后的新内容
                new_piece = full[prev_len:]
                print(new_piece, end="", flush=True)
                out_ans += new_piece
                prev_len = len(full)


            f.write(f'===== CASE {run_count} =====\n')
            f.write(f'Question\n: {q}\n')
            f.write(f'Claude-instant Answer\n: {out_ans}\n')
            f.write(f'Reference Answer\n: {a}\n\n')

            run_count += 1
        # f.close()
    parse_answer_file(args.output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args)