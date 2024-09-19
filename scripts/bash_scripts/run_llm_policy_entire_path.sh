export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1 2 3 4 5; do
    python3 scripts/llm_policy_entire_path.py \
        --env "Mario-8x11" \
        --variation $variation \
        --llm_model "gpt-3.5-turbo" \
        --add_text_desc "True" \
        --additional_expt_info ""
done