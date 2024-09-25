export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    python3 scripts/pddl_llm_modulo_plan.py \
        --llm_model "gpt-4o" \
        --variation $variation  \
        --num_agent_steps 20 \
        --num_backprompt_steps 5
done