export PYTHONPATH=$PYTHONPATH:$(pwd)

for expt_seed in 2; do
    python3 scripts/run_q_learning_baseline_household.py \
        --seed $expt_seed \
        --stochastic \
        --additional_info "_stochastic_flags"
done