export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1 2 3; do
    python scripts/store_reward_shaping_from_pddl.py \
        --env "Mario-8x11" \
        --variation $variation \
        --llm_model "gpt-4o" \
        --llm_plan "llm_modulo"
done