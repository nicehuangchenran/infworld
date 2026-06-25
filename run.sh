#!/bin/bash -l

conda activate inworld
echo $OMP_NUM_THREADS
python scripts/infworld_inference_input_oom.py 2>&1 | tee logs/online_train_$(date +%F_%H-%M-%S).log