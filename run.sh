#!/bin/bash

# 로그 저장 폴더
LOG_DIR=logs
mkdir -p ${LOG_DIR}

# 타임스탬프
TS=$(date +"%Y%m%d_%H%M%S")

# echo "===== START: $TS ====="

############################################
# 1. discuss_final_lvbench_stable.py
############################################
echo "[1] Running discuss_final_lvbench_stable.py"

# python discuss_final_lvbench_stable.py --save_path lvbench_anno_stable_logits0408_2.json --save_logits_path ./logits_0408_2  --return_logits \
python discuss_final_lvbench_stable.py --save_path lvbench_anno_stable_logits0409.json  \
    > ${LOG_DIR}/stable_${TS}.out \
    2> ${LOG_DIR}/stable_${TS}.err

if [ $? -eq 0 ]; then
    echo "[1] SUCCESS"
else
    echo "[1] FAILED"
fi