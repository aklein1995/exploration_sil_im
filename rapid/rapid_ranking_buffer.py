import numpy as np
from rapid.utils import Counter_Global
from gym.spaces import Discrete, Box

class RankingBuffer(object):
    def __init__(self, ob_space, ac_space, args):
        '''
        Args:
            w0: Weight for extrinsic rewards
            w1: Weight for local bonus
            w2: Weight for global bonus (sums of count-based exploration)
        '''
        print('\nobspace:',ob_space)
        print('obspace shape:',ob_space.shape)
        # determine space dim
        self.ob_shape = ob_space.shape
        self.ob_dim = 1
        for dim in self.ob_shape:
            self.ob_dim *= dim

        # determine action dim
        if isinstance(ac_space, Discrete):
            self.action_type = 'discrete'
        elif isinstance(ac_space, Box):
            self.action_type = 'box'
            self.ac_dim = ac_space.shape[0]
        else:
            ValueError('The action space is not supported.')

        self.size = args.buffer_size # max number of elements to be stored simultaneously
        self.data = None # buffer itself
        self.index = 0 # used to monitor number of elements in buffer
        self.score_type = args.score_type # 'discrete' or 'continious'
        self.counter = Counter_Global() # Class used to calculate visitation counts in global_score

        # weights related to score calculation
        self.w0 = args.w0
        self.w1 = args.w1
        self.w2 = args.w2
        print('Buffer weights:', self.w0, self.w1, self.w2)


    def insert(self, obs, acs, ret):

        # calculate local bonus
        if self.w1 > 0:
            local_bonus = get_local_bonus(obs, self.score_type)
        else:
            local_bonus = 0.0

        # expand dims to have the same as obs
        if self.action_type == 'discrete':
            _ac_data = np.expand_dims(acs, axis=1)
        elif self.action_type == 'box':
            _ac_data = acs


        # prepare data in the next shape:
        # [obs, act, episode_index, reward, local_bonus, score]
        # _data == new incoming experiences of a single episode
        num = obs.shape[0]
        _data = np.concatenate((
            obs.astype(float).reshape(num,-1),
            _ac_data,
            np.zeros((num,1)),
            np.expand_dims(np.repeat(ret,num), axis=1),
            np.expand_dims(np.repeat(local_bonus,num), axis=1),
            np.zeros((num,1)),
            ), axis=1)
        # shape in discrete minigrid: _data.shape --> [num_experiences,152] (147 of obs + 5 of others attributes)

        # update global counts and get episode index
        if self.w2 > 0:
            obs_to_calculate = _data[:, :self.ob_dim] # get all the stored observations
            episode_index = self.counter.add(obs_to_calculate)
        else:
            episode_index = 0

        # copy episode index 'num' times -- all the experiences are marked with the episode in which they were collected
        _data[:,-4] = np.repeat(episode_index,num)

        # Fill buffer
        if self.data is None:
            # if is empty
            self.data = _data
        else:
            # if not empty, concatenate the new data
            self.data = np.concatenate((self.data, _data), axis=0)

            # calculate global bonus
            if self.w2 > 0:
                # get current episode index
                episode_idx = self.data[:,-4].astype(int)
                # get updated global score of all the experiences stored in the buffer
                global_bonus = self.counter.get_bonus(episode_idx)
            else:
                global_bonus = 0.0
            # print('global bonus:',global_bonus)

            # SCORE: calculate bonus of the whole buffer with the updated global bonus
            self.data[:,-1] = (self.w0 * self.data[:,-3]) + (self.w1 * self.data[:,-2]) + (self.w2 * global_bonus)

            # Generate sorted idx based on the score
            sort_idx = self.data[:,-1].argsort()

            # Keep in buffer only at MAX self.size samples
            self.data = self.data[sort_idx][-self.size:]

        # update number of elements stored in the buffer
        self.index = self.data.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(range(0,self.index), batch_size)
        sampled_data = self.data[idx]
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)
        if self.action_type == 'discrete':
            acs = sampled_data[:,self.ob_dim].astype(int)
        elif self.action_type == 'box':
            acs = sampled_data[:,self.ob_dim:self.ob_dim+self.ac_dim]
        return obs, acs

    def reset(self):
        self.data = None
        self.index = 0

def get_local_bonus(obs, score_type):
    if score_type == 'discrete':
        obs = obs.reshape((obs.shape[0], -1))
        unique_obs = np.unique(obs, axis=0)
        total = obs.shape[0]
        unique = unique_obs.shape[0]
        score = float(unique) / total
    elif score_type == 'continious':
        obs = obs.reshape((obs.shape[0], -1))
        obs_mean = np.mean(obs, axis=0)
        score =  np.mean(np.sqrt(np.sum((obs - obs_mean) * (obs -obs_mean), axis=1)))
    else:
        raise ValueError('Score type {} is not defined'.format(score_type))

    return score
