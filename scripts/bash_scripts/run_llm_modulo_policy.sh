export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1 2 3 4 5 6 7 8 9 10
# gpt-3.5-turbo, gpt-4o-mini
for variation in 6 7 8 9 10; do
    python3 scripts/llm_modulo_policy.py \
        --env "Mario-8x11" \
        --variation $variation \
        --llm-model "gpt-3.5-turbo" \
        --add_text_desc "True" \
        --give_feasible_actions "True" \
        --give_tried_actions "True" \
        --num_agent_steps 500 \
        --num_backprompt_steps 5 \
        --additional_expt_info "Version5" \
        --prompt_version 5
done