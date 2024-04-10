export MODEL_DIR="/path/to/gligen-1-4"
export ADD_INS_EMBED=true
export open_time_embedding=learnable_frame
export open_box_attn=disable
export dynamic_size=disable
export enc_only=false
export dec_only=false
export injector_on=dec_fuse
export self_on=false # check false!!!
export track_query=false # check false!!!

accelerate launch train_for_vis.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=480 \
    --height=320 \
    --checkpointing_steps=10000 --checkpoints_total_limit=4 \
    --learning_rate=1e-5 --lr_warmup_steps=1000 --lr_scheduler cosine\
    --seed=1 \
    --mixed_precision="fp16" \
    --output_dir=./outputs \
    --pretrain_unet=/path/to/unet \
    --report_to=tensorboard \
    --num_frames=16 \
    --fg_reweight=2.0 \
    --cfg_name=youtube_cfg_480.py \
    --allow_tf32


