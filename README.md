# Towards Improving Exploration n Self-Imitation Learning using Intrinsic Motivation
This is the Tensorflow implementation of [IEEE Symposium Series on Computational Intelligence (SSCI)](https://ieeessci2022.org/) 2022 paper:
*Towards Improving Exploration n Self-Imitation Learning using Intrinsic Motivation*. 

We propose a simple method that combines the generation of intrinsic rewards with self-imitation learning techniques that rank previous episodes and replay them to reproduce good exploration behaviors through imitation (behavioral cloning). We built our method on top of [RAPID](https://arxiv.org/abs/2101.08152) and evaluate the performance over hard-exploration procedurally-generated environments on [MiniGrid](https://github.com/Farama-Foundation/MiniGrid). The results show that using intrinsic motivation techniques with self-imitation learning methods exhibits a equal or better performance and sample efficiency in comparison to execute those methods in isolation.

The implementation is based on [RAPID-github-repo](https://github.com/daochenzha/rapid.git), which also comprises the original implementation of [self-imitation-learning](https://github.com/junhyukoh/self-imitation-learning.git). 

## Cite This Work
```
Proceedings still not published.
```

## Installation
Please make sure that you have **Python 3.6** installed. First, clone the repo with
```
git clone https://github.com/aklein1995/exploration_sil_im.git
```
Then install the dependencies with **pip**:
```
pip3 install -r requirements.txt
```

## Example of use
The entry is `main.py`. Some important hyperparameters are as follows.
*   `--env`: what environment to be used
*   `--frames`: the number of frames/timesteps to be run
*   `--nsteps`: the time horizon selected for gathering experiences before a PPO update
*   `--log_dir`: the directory to save logs

More related to intrinsic motivation:
*   `--im_coef`: the intrinsic coefficient value (0=no IM)
*   `--im_type`: the intrinsic module/approach used to compute the intrinsic rewards
*   `--use_ep_counts`: boolean that is used to scale the generated rewards based on the episodic counts
*   `--use_1st_counts`: boolean that is used to just effectively reward the first time a state is reached in the scope of an episode

And more specifically to self-imitation-learning:
*   `--w0`: the weight of extrinsic reward score
*   `--w1`: the weight of local score
*   `--w2`: the weight of global score
*   `--buffer_size`: maximum number of experiences stored in the replay buffer
*   `--sl_until`: do the RAPID update until which timestep
*   `--disable_rapid`: use it to compare with PPO baseline or with PPO+intrinsic rewards


### Reproducing the result of MiniGrid environments
For example, if we want to just analyze the results with intrinsic motivation(i.e. bebold) but without RAPID in MultiRoom with 7 rooms of size 8, run
```
python3 main.py --log_dir MN7S8_ent001_im0005_bb_0 --seed 0 --disable_rapid --ent_coef 0.01 --im_type bebold --im_coef 0.005  --frames 20000000 --env 'MiniGrid-MultiRoom-N7-S8-v0'
```

On the other hand, the same experiment with RAPID and no intrinsic rewards, run
```
python3 main.py --log_dir MN7S8_ent001_w0w1w2_0 --seed 0 --ent_coef 0.01 --im_coef 0 --frames 20000000 --env 'MiniGrid-MultiRoom-N7-S8-v0'
```

To evaluate on differente environments, change  `--env` and configure the simulation based on your preferences.

