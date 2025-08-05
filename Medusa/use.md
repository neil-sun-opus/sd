python -u run_mmlu_with_medusa.py \
  --model-path /home/neil/sd/Medusa/ckpt/medusa-7b \
  --batch-size 8 \
  --shots 0 \
  --output results/medusa7b_mmlu_0shot.json

lm_eval \
  --model hf \
  --model_args pretrained=/home/neil/sd/Medusa/ckpt/medusa-7b,trust_remote_code=True,dtype=bfloat16 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/mmlu_medusa7b_local

python run_mmlu_open_source.py --ckpt_dir /home/neil/sd/Medusa/ckpt/medusa-7b

## greedy medusa 
CUDA_VISIBLE_DEVICES=0 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
## greedy medusa 2 
CUDA_VISIBLE_DEVICES=0 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
nohup env CUDA_VISIBLE_DEVICES=0 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py > a_medusa_greedy2.log 2>&1 &
## typical medusa
CUDA_VISIBLE_DEVICES=1 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py

nohup env CUDA_VISIBLE_DEVICES=1 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py > a_medusa_typical.log 2>&1 &

## typical vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
nohup env CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_vicuna-7b.py > a_vicuna_typical.log 2>&1 &

## greedy vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
nohup env CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_vicuna-7b.py > a_vicuna_greedy.log 2>&1 &

## greedy eagle
nohup env CUDA_VISIBLE_DEVICES=3 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_eagle3.py > a_eagle_greedy.log 2>&1 &

## greedy vicuna-13b-v1.3
CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
nohup env CUDA_VISIBLE_DEVICES=4 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_vicuna-13b.py > a_vicuna13b_greedy.log 2>&1 &


## greedy vicuna-7b
nohup env CUDA_VISIBLE_DEVICES=0 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_vicuna-7b-greedy.py > a_vicuna_greedy2.log 2>&1 &

## greedy medusa-7b
CUDA_VISIBLE_DEVICES=1 python /home/neil/sd/chain-of-thought-hub/MMLU/a_medusa_mmlu.py
nohup env CUDA_VISIBLE_DEVICES=1 python /home/neil/sd/chain-of-thought-hub/MMLU/a_medusa_mmlu.py > a_medusa_greedy.log 2>&1 &

## fast medusa-7b
nohup env CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/MMLU/a_medusa_mmlu_fast.py > a_medusa_fast.log 2>&1 &

## greedy vicuna-7b
nohup env CUDA_VISIBLE_DEVICES=3 python /home/neil/sd/chain-of-thought-hub/MMLU/a_vicuna_mmlu.py > a_vicuna_greedy.log 2>&1 &

## greedy eagle-7b
nohup env CUDA_VISIBLE_DEVICES=4 python /home/neil/sd/chain-of-thought-hub/MMLU/a_eagle_mmlu.py > a_eagle_greedy.log 2>&1 &

## greedy vicuna-13b
nohup env CUDA_VISIBLE_DEVICES=5 python /home/neil/sd/chain-of-thought-hub/MMLU/a_vicuna-13b_mmlu.py > a_vicuna-13b_greedy.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python /home/neil/sd/chain-of-thought-hub/gsm8k/a_medusa.py
CUDA_VISIBLE_DEVICES=0 python /home/neil/sd/Medusa/mini.py 


conda run -n medusa --no-capture-output \
  bash -c "cd /home/neil/sd/chain-of-thought-hub/gsm8k && \
           CUDA_VISIBLE_DEVICES=1 python a_medusa.py"

conda run -n medusa --no-capture-output \
  bash -c "cd /home/neil/sd/chain-of-thought-hub/MMLU && \
           CUDA_VISIBLE_DEVICES=1 python a_vicuna_mmlu.py"

CUDA_VISIBLE_DEVICES=3 python /home/neil/sd/chain-of-thought-hub/MMLU/a_medusa_mmlu.py

conda run -n medusa --no-capture-output \
  bash -c "cd /home/neil/sd/chain-of-thought-hub/gsm8k && \
           CUDA_VISIBLE_DEVICES=4 python a_vicuna-7b.py"

grep -E ' acc ' a_medusa_greedy.log > a_medusa_greedy.txt
grep -E ' acc ' a_vicuna_greedy.log > a_vicuna_greedy.txt
grep -E ' acc ' a_eagle_greedy.log > a_eagle_greedy.txt
grep -E ' acc ' a_vicuna-13b_greedy.log > a_vicuna-13b_greedy.txt
grep -E ' acc ' .log > .txt