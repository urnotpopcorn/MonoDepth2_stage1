CUDA_VISIBLE_DEVICES=6 \
    python3 train.py \
    --model_name stage1 \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --SIG 
