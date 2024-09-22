export PYTHONPATH=$PYTHONPATH:$(pwd)

for expt_seed in 1 2 3 4 5; do
    python3 scripts/run_q_learning_baseline_mario.py \
        --seed $expt_seed
done