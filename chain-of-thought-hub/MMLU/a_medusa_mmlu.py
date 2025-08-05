# evaluating Medusa model on MMLU

import re
import time
import json

import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from utils import *

from medusa.model.medusa_model import MedusaModel
import torch
from fastchat.model import get_conversation_template

model_id = 'vicuna-7b'
def build_prompt(question: str):
    conv = get_conversation_template(model_id)   # 拿到模板
    header = (
        "You are an expert assistant. "
        "After your reasoning, reply with the exact phrase "
        "\"the answer is\" followed by the capital letter (A, B, C or D)."
    )
    user_msg = f"{header}\n\n{question}"

    conv.append_message(conv.roles[0], user_msg) # user 消息
    conv.append_message(conv.roles[1], None)     # 预留 assistant 槽
    prompt = conv.get_prompt()                   # 拼接完整 prompt
    return prompt, conv

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

def main(tasks=TASKS):

    model = MedusaModel.from_pretrained(
        '/home/neil/sd/Medusa/ckpt/medusa-7b',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    mmlu_prompt = json.load(open('lib_prompt/mmlu-cot.json'))
    for task in tasks:
        print('Testing %s ...' % task)
        i = 0
        acc = 0
        task_data = load_dataset("lukaemon/mmlu", task, trust_remote_code=True)
        with open('outputs/medusa-7b-greedy/test_medusa_7b_vicuna_%s.txt' % task, 'w') as fd:
            for q_ in tqdm(task_data['test'], total=len(task_data['test'])):
                q = q_['input'] + '\n'
                for letter in ['A', 'B', 'C', 'D']:
                    q += '(' + letter + ') ' + q_[letter] + ' '
                # q += "\nA: Let's think step by step."  
                    
                prompt_q , _ = build_prompt(q)
                input_ids = tokenizer.encode(prompt_q, return_tensors="pt").to(
                    model.base_model.device
                )

                gen = model.medusa_generate(
                    input_ids,
                    temperature=0,
                    max_steps=250,
                    sampling='typical',
                    fast=False,
                )

                out_ans = ""
                prev_len = 0
                for chunk in gen:
                    full = chunk["text"]
                    # 取出上一次之后的新内容
                    new_piece = full[prev_len:]
                    print(new_piece, end="", flush=True)
                    out_ans += new_piece
                    prev_len = len(full)
                ans_, residual = extract_ans(out_ans)
                    
                a = q_['target']
                fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))
                i += 1
                
                if(test_answer_mmlu_(ans_, a)): acc += 1
            print('%s acc %.4f' % (task, acc / len(task_data['test'])))
    return 

if __name__ == '__main__':
    main()