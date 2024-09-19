import os
import argparse
import time
import datetime
import pickle
import tensorboardX
import sys

from algorithms.algos.a2c import A2CAlgo
from algorithms.algos.ppo import PPOAlgo

import utils
from utils import device
from utils.model import ACModel
import wandb
from utils.env_mario import *
# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "online"
WANDB_PROJECT = "neurips_24" # neurips_24, iclr_25

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=False, default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=False, default="Mario-8x11",
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--env_config_seed", required=False, type=int, default=0,
                    help="seed for environment configuration (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
# parser.add_argument("--expt_name", default=None, type=str,
#                     help="name of the experiment (default: {ALGO}_{ENV}_{ENV_CONFIG_SEED})")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--stochastic", default=False, action="store_true",
                    help="add stochastic actions with default probability of 0.9")
parser.add_argument("--llm_rs", default=False, action="store_true",
                    help="uses the stored llm-modulo policy for reward shaping")
parser.add_argument("--llm_variation", default=1, type=int,
                    help="variation of the llm policy to be used for reward shaping (default: 1). Allowed values: 1, 2, 3")
parser.add_argument("--additional_info", default='Experiment', type=str,
                    help="additional info to be added to model name for saving. E.g. - Baseline, RewardShaping, Text etc. (default: Experiment)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=5,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.0001,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'return_mean',
            'goal': 'maximize'
        },
        'parameters': {
            'algo': {
                'values': ['ppo']
            },
            'env': {
                'values': ['Mario-8x11']
            },
            'env_config_seed': {
                'values': [0]
            },

            'seed': {
                'values': [0]
            },
            'log_interval': {
                'values': [1]
            },
            'save_interval': {
                'values': [10]
            },
            'procs': {
                'values': [16]
            },
            'frames': {
                'values': [10**6]
            },            
            'additional_info': {
                'values': ['WANDB_sweep']
            },
            'llm_rs': {
                'values': [False]
            },
            
            'epochs': {
                'values': [4, 8, 16, 32]
            },
            'batch_size': {
                'values': [256, 512, 1024]
            },
            'frames_per_proc': {
                'values': [128, 256, 512, 1024]
            },
            'discount': {
                'values': [0.99, 0.95]
            },
            'lr': {
                'values': [0.001, 0.0001]
            },
            'gae_lambda': {
                'values': [0.95, 0.99]
            },
            'entropy_coef': {
                'values': [0.0001, 0.001]
            },
            'value_loss_coef': {
                'values': [0.5]
            },
            'max_grad_norm': {
                'values': [0.5]
            },
            'optim_eps': {
                'values': [1e-8]
            },
            'optim_alpha': {
                'values': [0.99]
            },
            'clip_eps': {
                'values': [0.2]
            },
            'recurrence': {
                'values': [1]
            },
            'text': {
                'values': [False]
            }
        },
    }
    
    # Initialize the wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)  
    
    def train_agent(config=None):
        with wandb.init(config=config):
            config = wandb.config        
            config.update(sweep_config, allow_val_change=True)

            # run the training script

            # config.model = f"{config.algo}/{config.env}/env_config_seed_{config.env_config_seed}/expt_seed_{config.seed}"
            if config.llm_rs:
                config.model = f"{config.additional_info}/LLM_Variation_{config.llm_variation}/{config.algo}/{config.env}/env_config_seed_{config.env_config_seed}/expt_seed_{config.seed}"
                expt_name = f"{config.additional_info}_LLM_Variation_{config.llm_variation}_{config.algo}_{config.env}_EnvSeed_{config.env_config_seed}"
            else:
                config.model = f"{config.additional_info}/{config.algo}/{config.env}/env_config_seed_{config.env_config_seed}/expt_seed_{config.seed}"
                expt_name = f"{config.additional_info}_{config.algo}_{config.env}_EnvSeed_{config.env_config_seed}"

            # wandb.init(project=WANDB_PROJECT,
            #            config=args,
            #            name=expt_name,
            #            dir=f"./storage/{config.model}",
            #            sync_tensorboard=True,
            #            )

            config.mem = config.recurrence > 1

            # Set run dir

            date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            default_model_name = f"{config.env}_{config.algo}_seed{config.seed}_{date}"

            model_name = config.model
            model_dir = utils.get_model_dir(model_name)

            # Load loggers and Tensorboard writer

            txt_logger = utils.get_txt_logger(model_dir)
            csv_file, csv_logger = utils.get_csv_logger(model_dir)
            tb_writer = tensorboardX.SummaryWriter(model_dir)

            # Log command and all script arguments

            txt_logger.info("{}\n".format(" ".join(sys.argv)))
            txt_logger.info("{}\n".format(args))

            # Set seed for all randomness sources

            utils.seed(config.seed)

            # Set device

            txt_logger.info(f"Device: {device}\n")

            # Load environments

            envs = [Env_Mario(info_img=True)]
            # for i in range(config.procs):
            #     envs.append(utils.make_env(env_key=config.env, seed=config.seed, stochastic=config.stochastic))

            txt_logger.info("Environments loaded\n")

            # Load training status

            try:
                status = utils.get_status(model_dir)
            except OSError:
                status = {"num_frames": 0, "update": 0}
            txt_logger.info("Training status loaded\n")

            # Load observations preprocessor

            #obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
            obs_space = np.array([8,11])
            preprocess_obss = None
            if "vocab" in status:
                preprocess_obss.vocab.load_vocab(status["vocab"])
            txt_logger.info("Observations preprocessor loaded")

            # Load model

            acmodel = ACModel(obs_space, envs[0].action_space, config.mem, config.text)
            if "model_state" in status:
                acmodel.load_state_dict(status["model_state"])
            acmodel.to(device)
            txt_logger.info("Model loaded\n")
            txt_logger.info("{}\n".format(acmodel))
            
            # load llm reward shaping plan
            if config.llm_rs:
                llm_rs_file = f"./storage/lm_modulo_visualization/{config.env}/seed_{config.env_config_seed}/variation_{config.llm_variation}/lm_modulo_policy_pbrs_nsrss.pkl"
                with open(llm_rs_file, 'rb') as f:
                    llm_rs_policy = pickle.load(f)
                txt_logger.info(f"LLM reward shaping plan loaded from {llm_rs_file}.\n")
            else:
                #llm_rs_policy = []
                llm_rs_policy = None

            # Load algo

            if config.algo == "a2c":
                algo = A2CAlgo(envs, acmodel, device, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                                        config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                                        config.optim_alpha, config.optim_eps, preprocess_obss, llm_rs_policy)
            elif config.algo == "ppo":
                algo = PPOAlgo(envs, acmodel, device, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                                        config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                                        config.optim_eps, config.clip_eps, config.epochs, config.batch_size, preprocess_obss, llm_rs_policy)
            else:
                raise ValueError("Incorrect algorithm name: {}".format(config.algo))

            if "optimizer_state" in status:
                algo.optimizer.load_state_dict(status["optimizer_state"])
            txt_logger.info("Optimizer loaded\n")

            # Train model

            num_frames = status["num_frames"]
            update = status["update"]
            start_time = time.time()

            while num_frames < config.frames:
                # Update model parameters
                update_start_time = time.time()
                exps, logs1 = algo.collect_experiences()
                logs2 = algo.update_parameters(exps)
                logs = {**logs1, **logs2}
                update_end_time = time.time()

                num_frames += logs["num_frames"]
                update += 1

                # Print logs

                if update % config.log_interval == 0:
                    fps = logs["num_frames"] / (update_end_time - update_start_time)
                    duration = int(time.time() - start_time)
                    return_per_episode = utils.synthesize(logs["return_per_episode"])
                    rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                    header = ["update", "frames", "FPS", "duration"]
                    data = [update, num_frames, fps, duration]
                    header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                    data += rreturn_per_episode.values()
                    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                    data += num_frames_per_episode.values()
                    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                    txt_logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                        .format(*data))

                    header += ["return_" + key for key in return_per_episode.keys()]
                    data += return_per_episode.values()

                    if status["num_frames"] == 0:
                        csv_logger.writerow(header)
                    csv_logger.writerow(data)
                    csv_file.flush()

                    for field, value in zip(header, data):
                        tb_writer.add_scalar(field, value, num_frames)

                # Save status

                if config.save_interval > 0 and update % config.save_interval == 0:
                    status = {"num_frames": num_frames, "update": update,
                            "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                    if hasattr(preprocess_obss, "vocab"):
                        status["vocab"] = preprocess_obss.vocab.vocab
                    utils.save_status(status, model_dir)
                    txt_logger.info("Status saved")
                    
                return_mean = data[4]
                    
                wandb.log({"return_mean": return_mean})
                    
    wandb.agent(sweep_id, function=train_agent, count=10)