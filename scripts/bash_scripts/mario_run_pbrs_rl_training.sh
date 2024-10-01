export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    for expt_seed in 8; do
        python3 scripts/run_q_learning_pbrs_mario.py \
            --seed $expt_seed \
            --stochastic \
            --additional_info "_stochastic_llm_vanilla_Variation_${variation}_eps_decay" \
            --reshape_reward True \
            --variation $variation \
            --llm_model "gpt-3.5-turbo" \
            --llm_plan "vanilla"
    done
done