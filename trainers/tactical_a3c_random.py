from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import tensorflow as tf
from base.base_train import BaseTrain
from pysc2.env import sc2_env
from pysc2.lib import actions as sc_actions
from utils.utilities import updateNetwork, addFeatureLayers, getAvailableActions, addGeneralFeatures, terminate


class TacticalTrainer(BaseTrain):
    def __init__(self, worker_id, config, sess, model, data, logger):
        super(TacticalTrainer, self).__init__(sess, model, data, config, logger)
        self.config = config

        self.name = worker_id
        self.episodeRewards = []
        self.session = sess
        self.number_of_actions = len(sc_actions.FUNCTIONS)
        self.val = 0

        # Tensorflow summary writer (for tensorboard)
        self.summaryWriter = tf.summary.FileWriter(os.path.join(self.config.summary_dir, self.config.map_name + "_" + self.config.test_id + "-" + self.name))
        self.screenSize = self.config.screen_size
        self.exploration = self.config.exploration
        self.mapName = self.config.map_name

        # Create local network
        self.localNetwork = model
        self.globalEpisodes = self.model.global_step_tensor
        self.increment = self.globalEpisodes.assign_add(1)
        self.episode_count = 0

        # define that when running a tf session with self.updatevars we want to update the worker to the global network
        self.updateVars = updateNetwork('global', self.name)

        # Setup sc environment
        game = sc2_env.SC2Env(
            map_name=self.config.map_name,
            step_mul=self.config.step_mul,
            visualize=False)

        self.env = game

    def work(self, thread_coordinator):
        self.episode_count = self.session.run(self.globalEpisodes)  # gets current global episode
        print("Starting worker '" + str(self.name) + "'")

        with self.session.as_default(), self.session.graph.as_default():
            while not thread_coordinator.should_stop():
                self.train_epoch()

    # Each episode
    def train_epoch(self):
        # Store values
        episode_values = []
        # Store Rewards
        episode_reward = 0
        # Is the minigame over?
        done = False

        # Reset minigame
        obs = self.env.reset()

        # each step
        while not done:
            # perform step, return exp
            exp, done, screen, action_info, obs = self.perform_env_action(obs)
            episode_values.append(exp[3])
            episode_reward += exp[2]

            if done:
                break

        # When done == true
        self.episodeRewards.append(episode_reward)
        print("Episode: " + str(self.episode_count) + " Reward: " + str(episode_reward))

        # Suppress stupid error.
        value_loss = 0
        policy_loss = 0
        variable_norms = 0

        # save model and statistics.
        if self.episode_count != 0:
            mean_reward = np.mean(self.episodeRewards[-1:])
            summary = tf.Summary()
            summary.value.add(tag='Reward', simple_value=float(mean_reward))
            self.summaryWriter.add_summary(summary, self.episode_count)
            self.summaryWriter.flush()  # flushes to disk

        if self.name == 'worker_0':  # TODO Maybe fix later.
            self.session.run(self.increment)

        self.episode_count += 1
        if self.episode_count >= self.config.total_episodes:
            terminate(self.config)

    def perform_env_action(self, obs):
        # add feature layers
        screen = addFeatureLayers(obs[0])

        # run session and get policies
        action_info = np.zeros([1, self.number_of_actions], dtype=np.float32)
        # list of available actions
        action_info[0, getAvailableActions(obs[0], self.mapName)] = 1

        gen_features, b_queue, selection = addGeneralFeatures(obs[0])

        action_policy = 0
        # Select action from policies
        action, action_exp, spatial_action, selected_action_array = self.select_action(action_policy,
                                                                                       obs[0],
                                                                                       screen,
                                                                                       gen_features,
                                                                                       b_queue,
                                                                                       selection)

        obs = self.env.step(action)  # Perform action on environment

        # Gets reward from current step
        reward = obs[0].reward
        # Check if the minigame has finished
        done = obs[0].last()

        # return experience
        return [screen, action_exp, reward, 0, spatial_action, gen_features, b_queue, selection, selected_action_array], done, \
               screen, action_info, obs,

    def select_action(self, action_policy, obs, screen, genFeatures, bQueue, selection):
        # returns list of chosen action intersected with pysc available actions (currently available actions)
        vActions = getAvailableActions(obs, self.mapName)
        # Pick an action with probabillity normActions(gets original probability from
        action_prob = np.random.choice(len(vActions))
        # validActions, not the normalized version) gives us action exploration
        act_id = vActions[action_prob]

        selected_action_array = []

        spaction = np.random.choice((self.config.screen_size ** 2), 1)
        spatial_action = [1, spaction]
        target = [int(spatial_action[1] // self.screenSize), int(spatial_action[1] % self.screenSize)]

        spatial_action[1] = target[0] * self.screenSize + target[1]

        # define second spatial action Todo find a suitable solution
        target2 = target[:]
        target2[0] = int(max(0, min(self.screenSize - 1, target[0] + 6)))
        target2[1] = int(max(0, min(self.screenSize - 1, target[1] + 6)))
        if act_id == sc_actions.FUNCTIONS.select_rect.id:
            target[0] = int(max(0, min(self.screenSize - 1, target[0] - 6)))
            target[1] = int(max(0, min(self.screenSize - 1, target[1] - 6)))

        # For experience
        action_exp = [act_id, vActions]

        act_args = []
        for arg in sc_actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap'):
                act_args.append([target[1], target[0]])
            elif arg.name in 'screen2':
                act_args.append([target2[1], target2[0]])
            elif arg.name in 'control_group_id':
                act_args.append([4])
            else:
                # No spatial action was used
                spatial_action[0] = 0
                act_args.append([0])
        return [sc_actions.FunctionCall(act_id, act_args)], action_exp, spatial_action, selected_action_array
