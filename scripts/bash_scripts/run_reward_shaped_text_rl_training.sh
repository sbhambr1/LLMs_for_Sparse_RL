export PYTHONPATH=$PYTHONPATH:$(pwd)

for env_cfg_seed in 0; do
    for expt_seed in 1 2 3 4 5; do
        python scripts/train.py \
            --algo "ppo" \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --env_config_seed $env_cfg_seed \
            --seed $expt_seed \
            --save-interval 10 \
            --frames 100000 \
            --stochastic \
            --text \
            --llm_rs \
            --llm_variation 1 \
            --additional_info "LLMRewardShapingText"
    done
done