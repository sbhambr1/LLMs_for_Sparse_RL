# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0
# 1 2 3 4 5 6 7 8 9 10

export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/evaluate_offline_rl.py \
    --env "MiniGrid-LavaGapS5-v0" \
    --model "ppo/MiniGrid-LavaGapS5-v0/env_config_seed_0/expt_seed_3/Baseline_Deterministic"