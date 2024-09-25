export PYTHONPATH=$PYTHONPATH:$(pwd)

for variation in 1; do
    python3 scripts/pddl_llm_vanilla_plan.py \
        --llm_model "gpt-4o" \
        --variation $variation    
done