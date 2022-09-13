import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from collections import deque
from gym.spaces import Discrete, Box


from rapid.baselines_utils.misc_utils import set_global_seeds, explained_variance
from rapid.baselines_utils.runners import AbstractEnvRunner
from rapid.baselines_utils import tf_util,logger

from rapid.rapid_ranking_buffer import RankingBuffer
from rapid.self_imitation import SelfImitation
from rapid.utils import safemean, sf01, constfn, Counter, BeBold ,limit_cuda_visible_devices

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, batch_size=256,
                buffer_size = 10000 ,gamma=0.99, sil_update=5, fn_reward=None, fn_obs=None,
                sil_value=0.01, sil_alpha=0.6, sil_beta=0.1,sil_loss_weight=0.1):
        sess = tf_util.make_session()

        # define models
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
        sil_model = policy(sess, ob_space, ac_space, None, None, reuse=True)
        # rapid_model = policy(sess, ob_space, ac_space, batch_size, 1, reuse=True)

        # Define PPO session..
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        ENTROPY_COEF = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        with tf.variable_scope('model'):
            params = tf.trainable_variables()

        # LOSS ON-POLICY (PPO)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # Trainer
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(train_rl, lr, cliprange, ent_c, obs, returns, masks, actions, values, neglogpacs, states=None):
            if not train_rl:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, ENTROPY_COEF:ent_c,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        # Self-Imitation learning
        self.sil = SelfImitation(sil_model.X, sil_model.vf,
                sil_model.entropy, sil_model.value, sil_model.neg_log_prob,
                ac_space, fn_reward=fn_reward, fn_obs=fn_obs,
                n_env=nbatch_act, batch_size=batch_size,
                n_update=sil_update, w_value=sil_value,
                w_entropy=ent_coef, gamma=gamma,
                max_steps=buffer_size, max_nlogp=100,
                alpha=sil_alpha, beta=sil_beta)

        self.sil.set_loss_weight(sil_loss_weight) # 0.1 for PPO; 1.0 in Atari with A2C
        self.sil.build_train_op(params, trainer, LR)

        def sil_train(cur_lr):
            return self.sil.train(sess, cur_lr)

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.sil_train = sil_train
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, ranking_buffer, nsteps, gamma, lam, batch_size, im_coef = 0, im_type = 'counts', use_episodic_counts = 0):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.episodes_count = 0
        self.ranking_buffer = ranking_buffer
        self.obs_buf = []
        self.acs_buf = []
        self.int_rews_buf = []
        self.batch_size = batch_size

        self.rapid_loss = np.nan
        self.sil_samples = np.nan
        self.sil_loss = np.nan

        self.im_coef = im_coef
        self.im_type = im_type
        self.use_episodic_counts = use_episodic_counts

        self.episodic_counter = Counter()

        if self.im_type=='bebold':
            print('Using BeBold')
            self.use_episodic_counts = 1
            self.intrinsic_counter = BeBold()
        else:
            print('Using Visitation Counts')
            self.intrinsic_counter = Counter()

    def reset_ranking_buffer(self):
        print('old buffer size:',self.ranking_buffer.index)
        self.ranking_buffer.reset()
        print('buffer reset')
        print('new buffer size:',self.ranking_buffer.index)

    def run(self, switch_off_im=0, do_buffer=False, do_sl=False, sl_num=5, lr=0.0001):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):

            # select action
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Environment step
            obs, rewards, self.dones, infos = self.env.step(actions)

            # add step to SIL
            self.model.sil.step(obs, actions, rewards, self.dones.copy()) # add only extrinsic reward

            # Intrinsic Motivation Reward related
            if (self.im_coef > 0) and (switch_off_im == 0):
                # get intrinsc bonus
                if self.im_type=='bebold':
                    rew_int = self.intrinsic_counter.compute_intrinsic_reward(obs=self.obs,nobs=obs)
                else:
                    rew_int = self.intrinsic_counter.compute_intrinsic_reward(obs)

                # scale if use episodic counts
                if self.use_episodic_counts:
                    # update visit counts for episodic counter
                    self.episodic_counter.update(obs)
                    # get number of times visited in episode
                    num_episodic_counts = self.episodic_counter.get_number_visits(obs)

                    if self.im_type=='bebold':
                        # generate mask
                        if num_episodic_counts == 1:
                            mask = 1
                        elif num_episodic_counts > 1:
                            mask = 0
                        else:
                            print('Error, 404 Not found')

                        rew_int *= mask
                    else:
                        # apply episodic counts regularization/scale
                        rew_int /= np.sqrt(num_episodic_counts)
                        # print('rew_int post episodic:',rew_int)

                    # if episode has finished...
                    if self.dones:
                        # reset episodic counts
                        self.episodic_counter = Counter()

                # combine with extrinsic reward
                rewards += self.im_coef*rew_int

            mb_rewards.append(rewards)
            if (self.im_coef > 0) and (switch_off_im == 0): self.int_rews_buf.append(rew_int)
            self.obs = np.copy(obs)

            for info in infos:
                maybeepinfo = info.get('episode')

                # if episode finished
                if maybeepinfo:
                    self.episodes_count += 1
                    # add int_rews_MonteCarloReturn
                    maybeepinfo['rint'] = np.sum(self.int_rews_buf)
                    epinfos.append(maybeepinfo)
                    self.obs_buf = []
                    self.acs_buf = []
                    self.int_rews_buf = []
                    # if do_sl:
                    #     # at self.imitation.py train function it executes n_update times the train step
                    #     self.sil_loss, sil_adv, self.sil_samples, sil_nlogp = self.model.sil_train(lr)


        # train/update the IM module
        for o in mb_obs:
            self.intrinsic_counter.update(o)
        self.intrinsic_counter.update(self.obs) # last s'

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


def learn_sil(policy, env, ranking_buffer, args,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, gpu_id=-1, optimal_score=1,
            txt_logger=None, csv_logger=None):

    # DEFINE gpus
    if gpu_id == -1:
        print('\n\nVisible gpu devices: All')
    else:
        limit_cuda_visible_devices(gpu_id)
        print('\n\nVisible gpu devices: ', gpu_id)

    # DEFINE loggers
    txt_logger = txt_logger
    csv_file = csv_logger[0]
    csv_logger = csv_logger[1]

    seed = args.seed
    batch_size = args.batch_size
    nsteps = args.nsteps
    total_timesteps = int(args.frames)#int(args.num_timesteps * 1.1)
    lr = args.lr
    im_coef = args.im_coef
    im_type = args.im_type
    use_episodic_counts = args.use_ep_counts

    ent_coef_now = ent_coef = args.ent_coef
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    buffer_size = args.buffer_size
    sil_loss_weight = args.sil_loss_weight
    sil_num_updates = args.sl_num

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, buffer_size = buffer_size,
                    sil_loss_weight= sil_loss_weight, sil_update = sil_num_updates , gamma=gamma)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, ranking_buffer=ranking_buffer,
                    nsteps=nsteps, gamma=gamma, lam=lam, batch_size=batch_size,
                    im_coef = im_coef, im_type = im_type, use_episodic_counts=use_episodic_counts)

    epinfobuf = deque(maxlen=100)
    tfirststart = duration =  time.time()
    nupdates = total_timesteps//nbatch

    sil_loss = np.nan
    sil_valid_samples = np.nan
    ############################################################################
    eprewmean = 0
    switch_off_im = 0
    decrease_entropy = 0
    only_update_onpolicy = 0
    only_update_offpolicy = 0
    reset_buffer = 0
    ############################################################################
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        ########################################################################
        # Collect rollout and also: fill ranking buffer & train rapid_loss
        ########################################################################
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(switch_off_im = switch_off_im,
                                                                                    do_buffer=False,
                                                                                    do_sl=True,
                                                                                    sl_num= args.sl_num,
                                                                                    lr = lrnow)
        epinfobuf.extend(epinfos)
        mblossvals = []

        ########################################################################
        # Train on-policy
        ########################################################################
        if (states is None) and (only_update_offpolicy == 0): # nonrecurrent version
            inds = np.arange(nbatch)
            # print('\nnum batch',nbatch)
            for _ in range(noptepochs):
                # print('Epoch')
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    # print('num train batches:',nbatch_train)
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(args.train_rl, lrnow, cliprangenow, ent_coef_now, *slices))

            # TRAIN SIL
            sil_loss, sil_adv, sil_samples, sil_nlogp = model.sil_train(lrnow)

        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:

            # get variable values
            frames = update*nsteps
            current_log_interval_duration = (tnow - tfirststart) - duration
            duration = tnow - tfirststart
            episodes = runner.episodes_count
            ev = explained_variance(values, returns)
            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            eplenmean =safemean([epinfo['l'] for epinfo in epinfobuf])
            eprewintmean =safemean([epinfo['rint'] for epinfo in epinfobuf])

            # losses related
            lossvals = np.mean(mblossvals, axis=0) #['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
            policy_loss = lossvals[0]
            value_loss = lossvals[1]
            entropy = lossvals[2]
            approxkl = lossvals[3]
            clipfrac = lossvals[4]
            rapid_loss = float(runner.rapid_loss)
            # sil_loss = float(runner.sil_loss)
            # sil_valid_samples = float(100*(runner.sil_samples/batch_size)) if not runner.sil_samples == np.nan else np.nan
            sil_valid_samples = float(100*(sil_samples/batch_size))

            # log data
            header = ["nupdates","total_timesteps","fps","duration","episodes","eprewmean","eprewintmean","eplenmean"]
            data = [update,frames,fps,current_log_interval_duration,episodes,eprewmean,eprewintmean,eplenmean]

            # losses & gradients
            header += ["entropy","policy_loss","value_loss","rapid_loss","sil_loss","sil_samples"]
            data += [entropy,policy_loss,value_loss,rapid_loss, sil_loss, sil_valid_samples]

            only_txt = data

            txt_logger.info(
            "U {} | F {} | FPS {:04.0f} | D {:.2f} | E {} | RΩ: {:.3f} | RintΩ: {:.3f} | Steps: {:.1f} | H {:.3f} | pL {:.3f} | vL {:.3f} | rL: {:.3f} | sL: {:.3f} | %Vsamples {} "
            .format(*only_txt))

            # overwrite csv file
            if update == 1:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            # overwrite tensorboardX
            # for field, value in zip(header, data):
            #     tb_writer.add_scalar(field, value, num_frames)
            """
            logger.logkv("serial_timesteps", (update*nsteps)/1e6)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", (update*nbatch)/1e6)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            # logger.logkv("return_int",returns_int)
            logger.logkv('eprewmean', eprewmean)
            logger.logkv('eplenmean', eplenmean)
            logger.logkv('time_elapsed', tnow - tfirststart)
            logger.logkv('episodes', runner.episodes_count)

            logger.logkv('switch_off_im',switch_off_im)
            logger.logkv('only_onpolicy',only_update_onpolicy)
            logger.logkv('only_offpolicy',only_update_offpolicy)
            logger.logkv('ent_coef',ent_coef_now)
            logger.logkv('lr',lrnow)
            logger.logkv('clipping',cliprangenow)

            logger.record_tabular("rapid_loss", float(runner.rapid_loss))
            logger.dumpkvs()
            """





        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model
