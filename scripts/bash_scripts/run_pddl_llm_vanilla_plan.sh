export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1 2 3; do
    python3 scripts/pddl_llm_vanilla_plan.py \
        --llm_model "gpt-3.5-turbo" \
        --variation $variation    
done