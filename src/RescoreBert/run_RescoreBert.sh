mode=$1
device=$2
dataset=$3
setting=$4
run_recog=false

CUDA_VISIBLE_DEVICES=${device} python ./train_RescoreBert.py ${mode}

# CUDA_VISIBLE_DEVICES=${device} python ./predict_RescoreBert.py ./checkpoint/ ${mode}