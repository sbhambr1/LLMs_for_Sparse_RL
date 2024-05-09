# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 0; do
        python3 scripts/llm_policy_new.py \
            --env "MiniGrid-DoorKey-6x6-v0" \
            --seed $seed \
            --variation $variation \
            --llm_model "gpt-3.5-turbo" \
            --add_text_desc "True" \
            --additional_expt_info "" \
            --num_agent_steps 30 
    done
done