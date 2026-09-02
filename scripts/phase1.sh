# Phase1 5090单卡训30k
python -u legged_lab/scripts/train_locomotion.py \
  --task=cola_phase_1_locomotion \
  --headless --num_envs=2048 --max_iterations=30000 \
  --logger=tensorboard --run_name=phase1_locomotion_1gpu