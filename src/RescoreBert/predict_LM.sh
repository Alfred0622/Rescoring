checkpoint_path=$1
lm=$2
cuda=$3

for i in $(seq 1 11)
do
    epoch=$((470*${i}))
    CUDA_VISIBLE_DEVICES=${cuda} python ./predict_${lm}.py ${checkpoint_path}/checkpoint-${epoch}/pytorch_model.bin epoch_${i}
done