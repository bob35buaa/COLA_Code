source /root/autodl-tmp/.sugar_deps/miniconda3/etc/profile.d/conda.sh
conda activate cola
cd /root/autodl-tmp/Workspace/COLA_Code
source setup_env.sh

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.sugar_deps/.cache/pip}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 \
  legged_lab/scripts/train_collaboration.py \
  --phase=3 \
  --distributed --headless --num_envs=2048 --max_iterations=16000 \
  --no_object_rank_count=1 --right_fixed_rank_count=1 \
  --resume=True \
  --load_run=/root/autodl-tmp/Workspace/COLA_Code/logs/cola_phase_2_teacher_left_fixed_bar/static_mix_phase2_teacher_4gpu \
  --checkpoint=model_7999.pt \
  --logger=tensorboard --run_name=phase3_student_4gpu
