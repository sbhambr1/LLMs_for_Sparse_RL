export PYTHONPATH=$PYTHONPATH:$(pwd)

for llm_model in "gpt-3.5-turbo" "gpt-4" "gpt-4o" "claude-3-haiku-20240307" "llama3-8b-instruct-v1:0"; do
    for plan in "vanilla" "llm_modulo"; do
        for variation in 4 5 6; do
            python scripts/store_reward_shaping_from_pddl.py \
                --env "Minecraft" \
                --variation $variation \
                --llm_model $llm_model \
                --llm_plan $plan
        done
    done
done