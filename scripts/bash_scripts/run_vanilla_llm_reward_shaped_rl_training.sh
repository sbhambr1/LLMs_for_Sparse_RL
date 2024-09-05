# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0 // 1000000 frames
# 1 2 3 4 5 6 7 8 9 10
# --additional_info "Baseline" "BaselineText" "LLMRewardShaping" "LLMRewardShapingText"

export PYTHONPATH=$PYTHONPATH:$(pwd)

for env_cfg_seed in 0; do
    for expt_seed in 1 2 3 4 5; do
        python scripts/train_vanilla_llm.py \
            --algo "ppo" \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --env_config_seed $env_cfg_seed \
            --seed $expt_seed \
            --save-interval 10 \
            --frames 100000 \
            --stochastic \
            --llm_rs \
            --llm_variation 0 \
            --llm_model "gpt-3.5-turbo" \
            --query_type "entire_path" \
            --additional_info "VanillaLLMpbrs"
    done
done