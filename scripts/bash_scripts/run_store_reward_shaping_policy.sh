export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0 2 3 5; do
    for variation in 1 2 3 4; do
        python scripts/store_reward_shaping_policy.py \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --seed $seed \
            --variation $variation \
            --same_rewards_same_states True \
            --llm_model "gpt-3.5-turbo"
    done
done