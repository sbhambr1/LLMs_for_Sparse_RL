export PYTHONPATH=$PYTHONPATH:$(pwd)

for seed in 0; do
    for variation in 1 2 3; do
        python scripts/store_reward_shaping_policy.py \
            --env "MiniGrid-DoorKey-5x5-v0" \
            --seed $seed \
            --variation $variation
    done
done