export PYTHONPATH=$PYTHONPATH:$(pwd)

for expt_seed in 1 2 3 4 5 6 7 8 9 10; do
    python3 scripts/run_q_learning_baseline_mario.py \
        --seed $expt_seed
done