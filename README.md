# Efficient Reinforcement Learning via Large Language Model-based Search

This repository contains the code for the paper "Efficient Reinforcement Learning via Large Language Model-based Search". The code is based on the PPO and A2C implementation on Minigrid environments from the [rl-starter-files](https://github.com/lcswillems/rl-starter-files) repository.

## Installation

```bash
conda create -n llm_modulo_sparse_rl python=3.10 --file requirements.yml
```

## Usage

### Baseline Experiments

1. Training baseline models (PPO or A2C)

    ```bash
    python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80000
    ```

2. Visualize agent's behavior

    ```bash
    python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
    ```

3. Evaluate agent's performance

    ```bash
    python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
    ```

### MEDIC-augmented LLM Experiments

Note, that you will have to use an OpenAI API key to run the experiments. The following can be run for one layout of the DoorKey-5x5 environment. Please change the `--env` and `--model` arguments to run the experiments on other environments. See bash scripts in the `scripts` directory for more examples.

1. Obtaining a plan for the relaxed problem:

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