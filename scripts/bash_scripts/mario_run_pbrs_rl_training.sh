export PYTHONPATH=$PYTHONPATH:$(pwd)
# variation 2, 4, 5
for variation in 5; do
    for expt_seed in 1 2 3 4 5; do
        python3 scripts/run_q_learning_pbrs_mario.py \
            --seed $expt_seed \
            --stochastic \
            --additional_info "_stochastic_llm_vanilla_Variation_${variation}" \
            --reshape_reward True \
            --variation $variation \
            --llm_model "gpt-3.5-turbo"
    done
done