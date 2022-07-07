# CUDA_VISIBLE_DEVICES=4 \
DATA_PATH="./data"
num_epoch=$1
resize_fac=$2
exp_name=$3
PY_ARGS=${@:4}

CUDA_VISIBLE_DEVICES=4 \
python train.py --resize_fac $1 --num_epoch $2 --exp_name $3 $PY_ARGS