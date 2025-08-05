import anthropic
import argparse
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, TypicalLogitsWarper
import torch
import os
from fastchat.model import get_conversation_template

parser = argparse.ArgumentParser()
parser.add_argument('--prompt_file', type=str, default='lib_prompt/prompt_original.txt')
parser.add_argument('--output_file', type=str, default='outputs/vicuna_7b_instant_gsm8k_test_greedy2.txt', help='Output file for claude-instant')


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
    
    local_model_path = '/home/neil/sd/models/vicuna-7b-v1.3'  # 本地权重目录


    load_path = local_model_path 
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(load_path)



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
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=1000,
                do_sample=False
            )
            out_ans = tokenizer.decode(
                outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            print('out_ans:', out_ans)


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