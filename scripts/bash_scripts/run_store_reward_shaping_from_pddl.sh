export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    python scripts/store_reward_shaping_from_pddl.py \
        --env "Mario-8x11" \
        --variation $variation \
        --llm_model "gpt-3.5-turbo" \
        --llm_plan "vanilla"
done