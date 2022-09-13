# from baselines import bench, logger
from rapid.baselines_utils.bench import Monitor
import numpy as np
import gym_minigrid
import gym
import os

logger_dir = 'trash'


def limit_cuda_visible_devices(gpu_id):
    # PhysicalDevice(name='/physical_device:XLA_CPU:0', device_type='XLA_CPU')
    # PhysicalDevice(name='/physical_device:XLA_GPU:0', device_type='XLA_GPU')
    # PhysicalDevice(name='/physical_device:XLA_GPU:1', device_type='XLA_GPU')
    # PhysicalDevice(name='/physical_device:XLA_GPU:2', device_type='XLA_GPU')
    # PhysicalDevice(name='/physical_device:XLA_GPU:3', device_type='XLA_GPU')
    # PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')
    # tf._api.v1.config
    # devices_in_computer = tf.config.experimental.list_physical_devices(device_type=None)
    # print('test2',devices_in_computer)

    # gpus = tf.config.list_physical_devices('GPU')
    # print('GPUs.',gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def _wrap_minigrid_env(env):
    from gym_minigrid.wrappers import ImgObsWrapper
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    # env = bench.Monitor(env, logger.get_dir())
    env = Monitor(env,logger_dir)
    return env

def make_env(env_id):
    ''' For multi-room environment
        Create the environment if it is not registered
    '''
    if env_id == 'MiniWorld-MazeS5-v0':
        from rapid.maze import MazeS5
        env = MazeS5()
        # env = bench.Monitor(env, logger.get_dir())
        env = Monitor(env,logger_dir)
        return env
    elif 'MiniGrid' in env_id:
        from gym_minigrid.register import env_list
        # Register the multi-room environment if it is not in env_list
        if env_id not in env_list:
            # Parse the string
            sp = env_id.split('-')
            room = int(sp[-3][1:])
            size = int(sp[-2][1:])

            # Create environment
            from gym_minigrid.envs import MultiRoomEnv
            class NewMultiRoom(MultiRoomEnv):
                def __init__(self):
                    super().__init__(
                        minNumRooms=room,
                        maxNumRooms=room,
                        maxRoomSize=size
                    )
            env = NewMultiRoom()
        else:
            env = gym.make(env_id)
        env = _wrap_minigrid_env(env)
        return env
    else: # Mujoco
        env = gym.make(env_id)
        # env = bench.Monitor(env, logger.get_dir())
        env = Monitor(env,logger_dir)

        return env

class Counter_Global(object):
    def __init__(self):

        # visitation counts
        self.counts = dict()
         # stores all observations of each episode
        self.episodes = dict()  # shape [num_samples, obs_dims*]
        # stores each episode's bonus --> {id_ep:bonus}
        self.episode_bonus = dict()
        # monitores how many episodes have been stored in the whole training
        self.episode_index = -1

    def add(self, obs):
        """
            Adds a visitation count for the input batch of observations
            -Updates the number of episodes
            -Saves in dictionary the observations
            -Updates the bonus of all the stores episodes in self.episodes
        """
        for ob in obs:
            ob = tuple(ob)
            if ob not in self.counts:
                self.counts[ob] = 1
            else:
                self.counts[ob] += 1
        self.episode_index += 1
        self.episodes[self.episode_index] = obs

        # after visitation counts updated, stores the same score for all the samples of an episode
        self.update_bonus()

        return self.episode_index

    def update_bonus(self):
        """
            Updates the episode bonus of all the stored experiences in self.episodes
        """
        for idx in self.episodes:
            bonus = []
            obs = self.episodes[idx]
            counter = 0
            # for each episode, update bonus
            for ob in obs:
                counter += 1
                ob = tuple(ob)
                count = self.counts[ob]
                bonus.append(count)
            bonus = 1.0 / np.sqrt(np.array(bonus))
            bonus = np.mean(bonus)
            self.episode_bonus[idx] = bonus

    def get_bonus(self, idxs):
        """
            Get bonus for all the experiences

            -Resize dictionaries of self.episodes and self.episode_bonus to only store info of episodes stored at the buffer
            # select only those episodes that are inside the idxs (that are taken from the buffer)
            # -Not all the episodes that have been visited in the train are stored! -- Not necessary, as the counts with which the
            bonus is calculated, is never resetted
        """

        self.episodes = {k:self.episodes[k] for k in idxs}
        self.episode_bonus = {k:self.episode_bonus[k] for k in idxs}

        bonus = []
        for idx in idxs:
            bonus.append(self.episode_bonus[idx])
        return np.array(bonus)

class Counter(object):
    def __init__(self):
        self.counts = {}

    def get_number_visits(self,obs):
        tup = self.encode_state(obs)
        return self.counts[tup]

    def compute_intrinsic_reward(self,obs):
        """
            Generates the Intrinsic reward bonus based on the encoded state/tuple
            -Accepts a single observation
        """
        tup = self.encode_state(obs)
        if tup in self.counts:
            return 1/np.sqrt(self.counts[tup])
        else:
            return 1

    def update(self,obs):
        """
            Add samples to the bins;
                -It is prepared to catch inputs of shape [batch_size, -1]
                -i.e. [2048,7,7,3]

            Returns: Intrinsic Rewards
        """
        tup = self.encode_state(obs)
        if tup in self.counts:
            self.counts[tup] += 1
        else:
            self.counts[tup] = 1

    def encode_state(self,state):
        """
            Encodes the state in a tuple or taking also into account the action
        """
        # print('data type:',type(state))
        # print(state.flatten().shape)
        state = state.flatten().tolist()
        return (tuple(state))

class BeBold(object):
    def __init__(self):
        self.counts = {}

    def get_number_visits(self,obs):
        tup = self.encode_state(obs)
        return self.counts[tup]

    def compute_intrinsic_reward(self,obs,nobs):
        """
            Generates the Intrinsic reward bonus based on the encoded state/tuple
            -Accepts a single observation
        """
        current_tup = self.encode_state(obs)
        next_tup = self.encode_state(nobs)

        if next_tup in self.counts:
            if current_tup in self.counts:
                return max((1/self.counts[next_tup]) - (1/self.counts[current_tup]),0)
            else:
                return max((1/self.counts[next_tup]) - 1, 0)
        else:
            if current_tup in self.counts:
                return max(1 - (1/self.counts[current_tup]) ,0)
            else:
                return 1

    def update(self,obs):
        """
            Add samples to the bins;
                -It is prepared to catch inputs of shape [batch_size, -1]
                -i.e. [2048,7,7,3]

            Returns: Intrinsic Rewards
        """
        tup = self.encode_state(obs)
        if tup in self.counts:
            self.counts[tup] += 1
        else:
            self.counts[tup] = 1

    def encode_state(self,state):
        """
            Encodes the state in a tuple or taking also into account the action
        """
        # print('data type:',type(state))
        # print(state.flatten().shape)
        state = state.flatten().tolist()
        return (tuple(state))
