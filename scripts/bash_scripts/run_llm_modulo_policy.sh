# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 0 1 2 3; do
        python3 scripts/llm_modulo_policy.py \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --seed $seed \
            --variation $variation \
            --llm-model "claude-3-haiku-20240307" \
            --add_text_desc "True" \
            --give_feasible_actions "True" \
            --give_tried_actions "True" \
            --additional_expt_info "" \
            --num_agent_steps 30 \
            --num_backprompt_steps 10
    done
done