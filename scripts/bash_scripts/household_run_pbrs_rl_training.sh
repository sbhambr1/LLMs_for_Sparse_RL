export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    for expt_seed in 5; do
        python3 scripts/run_q_learning_pbrs_household.py \
            --seed $expt_seed \
            --stochastic \
            --additional_info "_stochastic_llm_vanilla_Variation_${variation}" \
            --reshape_reward True \
            --variation $variation \
            --llm_model "gpt-3.5-turbo" \
            --llm_plan "vanilla"
    done
done