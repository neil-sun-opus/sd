#!/usr/bin/env bash
# exit on syntax errors, keep pipefail, but *do not* stop on non‑zero exit codes of commands
set -o pipefail

#---------------------------------------- 基本配置 ----------------------------------------
# true 只打印、不执行；可通过 "DRY_RUN=true bash run.sh" 覆盖
DRY_RUN=${DRY_RUN:-false}
BASE1="python test.py"
BASE2="python test_p2.py"

TEMPS=(0   0.7)     # temperature
TOPPS=("" 0.9)      # top_p；空串表示不加 --top_p

#---------------------------------------- 模型与 ckpt --------------------------------------
declare -A CKPT_TABLE
CKPT_TABLE[vicuna]="vicuna-7b-v1.3 vicuna-13b-v1.3"
CKPT_TABLE[medusa]="medusa-vicuna-7b-v1.3 medusa-vicuna-13b-v1.3"

EA_CKPTS=(  "EAGLE3-Vicuna1.3-13B" )
EAGLE_CKPTS=(  "vicuna-13b-v1.3" )

#---------------------------------------- 运行 ---------------------------------------------
mkdir -p logs

for MODEL in vicuna medusa eagle; do
  if [[ "$MODEL" != "eagle" ]]; then
    for CKPT in ${CKPT_TABLE[$MODEL]}; do
      for idx in "${!TEMPS[@]}"; do
        T="${TEMPS[$idx]}"
        P="${TOPPS[$idx]}"

        for BASE in "$BASE1" "$BASE2"; do
          CMD="$BASE --use_template --model $MODEL --model_ckpt ../model_ckpt/$CKPT --temperature $T"
          [[ -n "$P" ]] && CMD+=" --top_p $P"

          # 抽出命令行里最后一个单词作为脚本名（test.py / test_p2.py）
          script_short=$(basename "$(echo "$BASE" | awk '{print $NF}')")
          LOG="logs/$(date +%Y%m%d_%H%M%S)_${MODEL}_$(basename "$CKPT")_T${T}_P${P:-none}_${script_short}.log"
          echo ">> $CMD  > $LOG 2>&1 &"

          if ! $DRY_RUN; then
            echo "└─ executing..."
            eval "$CMD" >"$LOG" 2>&1
          fi
        done
      done
    done

  else
    for idx in "${!EA_CKPTS[@]}"; do
      CKPT="${EAGLE_CKPTS[$idx]}"
      EA="${EA_CKPTS[$idx]}"

      for idx2 in "${!TEMPS[@]}"; do
        T="${TEMPS[$idx2]}"
        P="${TOPPS[$idx2]}"

        for BASE in "$BASE1" "$BASE2"; do
          CMD="$BASE --use_template --model eagle --model_ckpt ../model_ckpt/$CKPT --ea_ckpt ../model_ckpt/$EA --temperature $T"
          [[ -n "$P" ]] && CMD+=" --top_p $P"

          script_short=$(basename "$(echo "$BASE" | awk '{print $NF}')")
          LOG="logs/$(date +%Y%m%d_%H%M%S)_eagle_$(basename "$CKPT")_EA$(basename "$EA")_T${T}_P${P:-none}_${script_short}.log"
          echo ">> $CMD  > $LOG 2>&1 &"

          if ! $DRY_RUN; then
            echo "└─ executing..."
            eval "$CMD" >"$LOG" 2>&1
          fi
        done
      done
    done
  fi
done

echo "全部任务执行完毕（顺序模式）。DRY_RUN=$DRY_RUN"