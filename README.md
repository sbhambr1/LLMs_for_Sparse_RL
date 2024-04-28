# Reward Shaping using Large Language Models for Text-based Reinforcement Learning

This repository contains the code for the paper "Reward Shaping using Large Language Models for Text-based Reinforcement Learning". The code is based on the PPO and A2C implementation on Minigrid environments from the [rl-starter-files](https://github.com/lcswillems/rl-starter-files) repository.

## Installation

@TODO: Update requirements.txt

```bash
git clone https://github.com/sbhambr1/llm_modulo_sparse_rl
cd llm_modulo_sparse_rl
python3 -m venv venv
pip install -r requirements.txt
```

### Tensorboard Aggregator (Optional)

To use the tensorboard aggregator, clone [this](https://github.com/Spenhouet/tensorboard-aggregator) repository and install the requirements.

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

### LLM-Modulo Experiments (W.I.P.)

#### Important files

1. `scripts/llm_policy.py`: Generate the LLM policy directly from the LLM model.

2. `scripts/manual_policy.py`: Generate the LLM policy that can be used by the RL agent by manually specifying the actions.

3. `scripts/vlm_policy.py`: Generate the VLM policy directly from the VLM model.

4. `llm_modulo/llm_modulo.py`: Contains the LLM-Modulo implementation that contains critic/verifier functions for each (env, seed) pair.

5. `llm_modulo/env_constraints.py`: Contains the environment constraints for each environment. Modify to add constraints for new environments and seeds.
