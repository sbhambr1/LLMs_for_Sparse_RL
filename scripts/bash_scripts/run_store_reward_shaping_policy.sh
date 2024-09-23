export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1 2 3 4 5; do
    python scripts/store_reward_shaping_policy.py \
        --env "Mario-8x11" \
        --variation $variation \
        --llm_model "gpt-3.5-turbo" \
        --same_rewards_same_states True
done