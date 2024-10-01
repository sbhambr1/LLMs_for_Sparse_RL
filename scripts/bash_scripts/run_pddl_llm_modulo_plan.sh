export PYTHONPATH=$PYTHONPATH:$(pwd)

for llm_model in "gpt-3.5-turbo" "gpt-4" "gpt-4o" "claude-3-haiku-20240307" "meta.llama3-8b-instruct-v1:0"; do
    for variation in 11 12 13 14 15; do
        python3 scripts/pddl_llm_modulo_plan.py \
            --llm_model $llm_model \
            --variation $variation  \
            --num_agent_steps 20 \
            --num_backprompt_steps 5
    done
done