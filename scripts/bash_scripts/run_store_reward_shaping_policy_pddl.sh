export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 3; do
    python scripts/store_reward_shaping_from_pddl.py \
        --env "Household" \
        --variation $variation \
        --llm_model "gpt-4o" \
        --llm_plan "llm_modulo"
done