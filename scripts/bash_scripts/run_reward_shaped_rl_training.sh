# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5, variations 1 2 3 4
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2, variations 1 2 3
# MiniGrid-LavaGapS5-v0, seeds= 0, variations 1 2 3
# MiniGrid-DoorKey-6x6-v0, seeds= 0, variations 1 2 3 4, frames 1000000

export PYTHONPATH=$PYTHONPATH:$(pwd)

for env_cfg_seed in 0; do
    for expt_seed in 1 2 3 4 5 6 7 8 9 10; do
        for llm_var in 1 2 3 4; do
            python scripts/train.py \
                --algo "ppo" \
                --env "MiniGrid-DoorKey-6x6-v0" \
                --env_config_seed $env_cfg_seed \
                --seed $expt_seed \
                --save-interval 10 \
                --frames 1000000 \
                --stochastic \
                --llm_rs \
                --llm_variation $llm_var \
                --additional_info "pbrsLLMRewardShaping"
        done
    done
done