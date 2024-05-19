# env = 
# MiniGrid-DoorKey-5x5-v0, seeds= 0 2 3 5
# MiniGrid-Empty-Random-5x5-v0, seeds= 0 1 2 
# MiniGrid-LavaGapS5-v0, seeds= 0
# MiniGrid-DoorKey-6x6-v0, seeds= 0

export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 1 2 3; do
        python scripts/store_reward_shaping_policy_ablation.py \
            --env "MiniGrid-Empty-Random-5x5-v0" \
            --seed $seed \
            --variation $variation
    done
done