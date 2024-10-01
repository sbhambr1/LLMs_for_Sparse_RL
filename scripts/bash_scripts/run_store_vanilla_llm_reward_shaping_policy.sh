# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0

# models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o, claude-3-haiku-20240307 (small), claude-3-sonnet-20240229 (medium), claude-3-opus-20240229 (large), meta.llama3-8b-instruct-v1:0, meta.llama3-1-8b-instruct-v1:0


export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 1 2 3 ; do
        python scripts/store_vanilla_llm_reward_shaping_policy.py \
            --env "MiniGrid-Empty-Random-5x5-v0" \
            --seed $seed \
            --variation $variation \
            --llm_model "meta.llama3-8b-instruct-v1:0" \
            --same_rewards_same_states True \
            --query_type "entire_path"  
    done
done