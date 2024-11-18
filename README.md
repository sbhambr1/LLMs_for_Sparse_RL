# Extracting Heuristics from Large Language Models for Reward Shaping in Reinforcement Learning

This repository contains the code for the paper "Extracting Heuristics from Large Language Models for Reward Shaping in Reinforcement Learning". The code is based on the PPO and A2C implementation on Minigrid environments from the [rl-starter-files](https://github.com/lcswillems/rl-starter-files) repository, and the Q-learning implementation on Mario, Minecraft, and Household, from [ASGRL](https://github.com/GuanSuns/ASGRL/tree/main).

## Installation

```bash
conda create -n llm_modulo_sparse_rl python=3.10 --file requirements.yml
```

## Usage

### Baseline Experiments

- Training baseline models (Q-learning)

```bash
./scripts/bash_scripts/minecraft_run_baseline_rl_training.sh
```

### LLM Experiments

Note, that you will have to use an API key to run the experiments for the respective LLMs.

- Obtaining a hierarchical relaxed plan with direct LLM prompting:

```bash
./scripts/bash_scripts/run_pddl_llm_vanilla_plan.sh
```

- Obtaining a hierarchical relaxed plan with verifier-augmented LLM prompting:

```bash
./scripts/bash_scripts/run_pddl_llm_modulo_plan.sh
```

### Verifier-augmented LLM Experiments

Note, that you will have to use an OpenAI API key to run the experiments. The following can be run for one layout of the DoorKey-5x5 environment. Please change the `--env` and `--model` arguments to run the experiments on other environments. See bash scripts in the `scripts` directory for more examples.

1. Obtaining a plan for the deterministic relaxed problem:

    ```bash
    ./scripts/bash_scripts/run_llm_modulo_policy.sh
    ```

2. Construct and store the reward shaping function:

    ```bash
    ./scripts/bash_scripts/run_store_reward_shaping_policy.sh
    ```

3. Training the agent with the reward shaping function:

    ```bash
    ./scripts/bash_scripts/run_reward_shaped_rl_training.sh
    ```