# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for env_cfg_seed in 0 2 3 5; do
    for expt_seed in 1 2 3 4 5; do
        for llm_var in 1 2 3 4; do
            python scripts/train.py \
                --algo "ppo" \
                --env "MiniGrid-DoorKey-5x5-v0" \
                --env_config_seed $env_cfg_seed \
                --seed $expt_seed \
                --save-interval 10 \
                --frames 100000 \
                --stochastic \
                --llm_rs \
                --llm_variation $llm_var \
                --additional_info "LLMpbrs_nsrss"
        done
    done
done