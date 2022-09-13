# from baselines.a2c.policies import CnnPolicy
# from baselines.common.vec_env.vec_normalize import VecNormalize
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from rapid.baselines_utils.vec_env import VecNormalize, DummyVecEnv
from rapid.rapid_agent import learn_rapid
from rapid.sil_agent import learn_sil
from rapid.utils import make_env
from rapid.models import MlpPolicy,MlpPolicy_unique,CnnPolicy
from rapid.rapid_ranking_buffer import RankingBuffer

def train(args, gpu_id, txt_logger, csv_logger, optimal_score):

    # exploration score type
    if 'MiniGrid' in args.env:
        args.score_type = 'discrete'
        args.train_rl = True
        if args.use_sharednetwork:
            policy_fn = MlpPolicy_unique
        else:
            policy_fn = MlpPolicy
    elif args.env == 'MiniWorld-MazeS5-v0':
        args.score_type = 'continious'
        args.train_rl = True
        policy_fn = CnnPolicy
    else: # MuJoCo
        args.score_type = 'continious'
        if args.disable_rapid:
            args.train_rl = True
        else:
            args.train_rl = False

        if args.use_sharednetwork:
            policy_fn = MlpPolicy_unique
        else:
            policy_fn = MlpPolicy

    # Make the environment
    def _make_env():
        env = make_env(args.env)
        env.seed(args.seed)
        return env

    env = DummyVecEnv([_make_env])
    if not 'MiniGrid' in args.env and not args.env == 'MiniWorld-MazeS5-v0': # Mujoco
        env = VecNormalize(env)


    # Initialize the buffer
    ranking_buffer = RankingBuffer(ob_space=env.observation_space,
                                   ac_space=env.action_space,
                                   args=args)

    # Start training -- Select Self-Imitation-Learning or RAPID (no imitation learning schema is allowed inside RAPID setting w0 w1 and w2 = 0)
    if args.use_sil:
        learn_sil(policy = policy_fn, env = env, ranking_buffer = ranking_buffer, args = args, gpu_id = gpu_id , txt_logger=txt_logger, csv_logger = csv_logger, optimal_score = optimal_score)
    else:
        learn_rapid(policy = policy_fn, env = env, ranking_buffer = ranking_buffer, args = args, gpu_id = gpu_id , txt_logger=txt_logger, csv_logger = csv_logger, optimal_score = optimal_score)

    env.close()
