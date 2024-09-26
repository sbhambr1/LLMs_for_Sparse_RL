export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    for expt_seed in 1 2; do
        python3 scripts/run_q_learning_pbrs_craft.py \
            --seed $expt_seed \
            --stochastic \
            --additional_info "_stochastic_llm_vanilla_Variation_${variation}" \
            --reshape_reward True \
            --variation $variation \
            --llm_model "gpt-4o" \
            --llm_plan "llm_modulo"
    done
done