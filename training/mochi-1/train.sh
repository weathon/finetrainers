#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

DATA_ROOT="videos_prepared"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-lora"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python text_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --pin_memory \
  --max_train_steps 2000 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --enable_model_cpu_offload \
  --optimizer adamw \
  --validation_prompt \"There is a *crab* blending into a +rocky ocean floor+ where the crab’s mottled brown shell, rough texture, and uneven shape closely match the scattered rocks and coarse sand, all in muted brown and grey tones. The crab moves slowly and subtly, making it difficult to distinguish as its rough brown pattern looks just like a piece of rock among the uneven, similarly colored stones and patches of sand.\" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_steps 100 \
  --checkpointing_steps 100 \
  --allow_tf32 \
  --report_to wandb \
  --push_to_hub"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
