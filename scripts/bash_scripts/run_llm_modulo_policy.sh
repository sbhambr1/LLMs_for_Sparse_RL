export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 1 2 3; do
        python3 scripts/llm_modulo_policy.py \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --seed $seed \
            --variation $variation \
            --llm-model "gpt-3.5-turbo" \
            --add_text_desc "True" \
            --give_feasible_actions "True" \
            --give_tried_actions "True" \
            --additional_expt_info "" \
            --num_agent_steps 30 \
            --num_backprompt_steps 20
    done
done