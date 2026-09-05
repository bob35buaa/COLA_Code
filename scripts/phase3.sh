# Phase3 5090单卡：left-fixed-bar student（非 static-mix）
conda activate cola
source setup_env.sh

python -u legged_lab/scripts/train_collaboration.py \
  --phase=3 \
  --headless --num_envs=2048 --max_iterations=30000 \
  --resume=True \
  --load_run=/home/ubuntu/Workspace/COLA_Code/logs/cola_phase_2_teacher_left_fixed_bar/static_mix_phase2_teacher_4gpu \
  --checkpoint=model_7999.pt \
  --logger=tensorboard --run_name=phase3_student_1gpu
