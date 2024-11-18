export PYTHONPATH=$PYTHONPATH:$(pwd)

for expt_seed in 20; do
    python3 scripts/run_q_learning_baseline_craft.py \
        --seed $expt_seed \
        --stochastic \
        --additional_info "_stochastic_flags"
done