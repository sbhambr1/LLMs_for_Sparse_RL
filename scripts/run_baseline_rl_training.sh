# env = 
# MiniGrid-DoorKey-5x5-v0, 
# MiniGrid-Empty-Random-5x5-v0, 
# MiniGrid-LavaGapS5-v0, 
# MiniGrid-KeyCorridorS3R1-v0,
# MiniGrid-DoorKey-6x6-v0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for env_cfg_seed in 0 2 3 5; do
    for expt_seed in 1 2 3 4 5 6 7 8 9 10; do
        python scripts/train.py \
            --algo "ppo" \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --env_config_seed $env_cfg_seed \
            --seed $expt_seed \
            --save-interval 10 \
            --frames 100000 \
            --stochastic \
            --additional_info "Baseline"
    done
done