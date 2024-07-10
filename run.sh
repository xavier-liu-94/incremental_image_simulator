MODELSCOPE_CACHE=/media/xavier/Samsumg/.cache/modelscope/hub accelerate launch experiment.py   \
    --output_dir="./output_example5" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=30000 