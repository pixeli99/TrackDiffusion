accelerate launch train_svd_tracklet.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --checkpointing_steps=1000 --checkpoints_total_limit=3 \
    --learning_rate=2e-4 --lr_warmup_steps=750 \
    --seed=1 \
    --mixed_precision="fp16" \
    --validation_steps=9999999 \
    --train_mode="stage1"