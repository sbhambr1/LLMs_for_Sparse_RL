# env = 
# MiniGrid-DoorKey-5x5-v0, 
# MiniGrid-Empty-Random-5x5-v0, 
# MiniGrid-LavaGapS5-v0, 
# MiniGrid-KeyCorridorS3R1-v0,
# MiniGrid-DoorKey-6x6-v0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 1; do
    for variation in 1 2 3 4 5 6 7 8 9 10; do
        python scripts/llm_modulo_policy.py \
            --env "MiniGrid-LavaGapS5-v0" \
            --seed $seed \
            --variation $variation \
            --llm-model "gpt-3.5-turbo" \
            --add_text_desc "True" \
            --give_feasible_actions "True" \
            --give_tried_actions "True" \
            --additional_expt_info "" \
            --num_agent_steps 50 \
            --num_backprompt_steps 20
    done
done