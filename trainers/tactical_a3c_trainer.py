from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
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
        self.episodeMeans = []
        self.session = sess
        self.number_of_actions = len(sc_actions.FUNCTIONS)
        self.experience_buffer = []
        self.val = 0

        # Tensorflow summary writer (for tensorboard)
        self.summaryWriter = tf.summary.FileWriter(self.config.summary_dir + "/" + self.config.map_name + "_" +self.config.test_id + "-" + self.name)
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

    def train_step(self):

        experience_buffer = np.array(self.experience_buffer)

        # Splitting of experience
        observations = experience_buffer[:, 0]
        actions = experience_buffer[:, 1]
        rewards = experience_buffer[:, 2]
        values = experience_buffer[:, 3]
        spatial_action = experience_buffer[:, 4]
        gen_features = experience_buffer[:, 5]
        obs_build = experience_buffer[:, 6]
        selections = experience_buffer[:, 7]
        # stores available actions for each timestep
        action_infos = []

        # Spatial and non spatial action preparation
        buffer_size = len(experience_buffer)

        # used for storing whether the current action made use of a spatial action
        valid_spatial_action = np.zeros([buffer_size], dtype=np.float32)
        # stores the picked spatial action(i[1]) for each experience tuple
        selected_spatial_action = np.zeros([buffer_size, self.screenSize ** 2], dtype=np.float32)
        # stores which actions were valid at a given time
        valid_actions = np.zeros([buffer_size, self.number_of_actions], dtype=np.float32)
        # stores which action was selected at a given time
        selected_action = np.zeros([buffer_size, self.number_of_actions], dtype=np.float32)

        value_target = np.zeros([buffer_size], dtype=np.float32)

        lr = self.config.learning_rate * (1 - 0.5 * self.episode_count / self.config.total_episodes)

        if math.isnan(self.val):
            value_target[-1] = 0
        else:
            value_target[-1] = self.val

        # goes through each timestep in experience buffer
        for t in range(0, buffer_size):
            # was a spatial action used during this timestep
            if spatial_action[t][0] == 1:
                # set spatial action chosen during current timestep
                selected_spatial_action[t, spatial_action[t][1]] = 1
                # a spatial action was used during this timestep
                valid_spatial_action[t] = 1

            # set which actions were valid during this timestep
            valid_actions[t, actions[t][1]] = 1
            # set action chosen during this timestep
            selected_action[t, actions[t][0]] = 1
            # stores valid actions for a specific timestep
            action_info = np.zeros([1, self.number_of_actions], dtype=np.float32)
            # sets valid actions for current timestep
            action_info[0, actions[t][1]] = 1
            action_infos.append(action_info)

        for t in range(buffer_size - 2, -1, -1):
            value_target[t] = rewards[t] + self.config.gamma * value_target[t + 1]

        # changes dimensions from [buffersize, 1, NB_actions] to [buffersize, NB_actions]
        action_infos = np.concatenate(action_infos, axis=0)

        # Define feed to use for Updating the global network using gradients from loss
        feed_dict = {self.localNetwork.valueTarget: value_target,
                     self.localNetwork.screen: np.vstack(observations),
                     self.localNetwork.actionInfo: action_infos,
                     self.localNetwork.validActions: valid_actions,
                     self.localNetwork.selectedAction: selected_action,
                     self.localNetwork.selectedSpatialAction: selected_spatial_action,
                     self.localNetwork.validSpatialAction: valid_spatial_action,
                     self.localNetwork.learningRate: lr,
                     self.localNetwork.generalFeatures: np.vstack(gen_features),
                     self.localNetwork.buildQueue: np.vstack(obs_build),
                     self.localNetwork.selection: np.vstack(selections),
                     }

        # Generate statistics from our network to periodically save and start the network feed
        value_loss, policy_loss, variable_norms, _ = self.session.run([self.localNetwork.valueLoss,
                                                                       self.localNetwork.policyLoss,
                                                                       # self.localNetwork.grads,
                                                                       self.localNetwork.varNorms,
                                                                       self.localNetwork.applyGrads],
                                                                      feed_dict=feed_dict)
        # Returns statistics for our summary writer
        return value_loss / len(experience_buffer), policy_loss / len(experience_buffer), variable_norms

    def work(self, thread_coordinator):
        self.episode_count = self.session.run(self.globalEpisodes)  # gets current global episode
        print("Starting worker '" + str(self.name) + "'")

        with self.session.as_default(), self.session.graph.as_default():
            while not thread_coordinator.should_stop():
                self.train_epoch()


    # Each episode
    def train_epoch(self):

        # reset local network to global network (updateVars = vars of global network)
        self.session.run(self.updateVars)
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
            self.experience_buffer.append(exp)

            episode_values.append(exp[3])
            episode_reward += exp[2]

            if len(self.experience_buffer) >= self.config.buffer_size and not done:
                # we don't know what our final return is, so we bootstrap from our current value estimation.
                self.val = self.session.run(self.localNetwork.value,
                                            feed_dict={self.localNetwork.screen: screen,
                                                       self.localNetwork.actionInfo: action_info,
                                                       self.localNetwork.generalFeatures: self.experience_buffer[-1][5],
                                                       self.localNetwork.buildQueue: self.experience_buffer[-1][6],
                                                       self.localNetwork.selection: self.experience_buffer[-1][7],
                                                       })[
                    0]
                value_loss, policy_loss, variable_norms = self.train_step()
                self.experience_buffer = []
                self.session.run(self.updateVars)
            if done:
                break

        # When done == true
        self.episodeRewards.append(episode_reward)
        self.episodeMeans.append(np.mean(episode_values))
        print("Episode: " + str(self.episode_count) + " Reward: " + str(episode_reward))

        # Suppress stupid error.
        value_loss = 0
        policy_loss = 0
        variable_norms = 0

        # Update the network using the experience buffer at the end of the episode.
        if len(self.experience_buffer) != 0:
            value_loss, policy_loss, variable_norms = self.train_step()

        # save model and statistics.
        if self.episode_count != 0:
            # makes sure only one of our workers saves the model
            if self.episode_count % 20 == 0 and self.name == 'worker_0':
                self.localNetwork.save(self.session)

            mean_reward = np.mean(self.episodeRewards[-1:])
            mean_value = np.mean(self.episodeMeans[-1:])
            summary = tf.Summary()
            summary.value.add(tag='Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Value', simple_value=float(mean_value))
            summary.value.add(tag='Value Loss', simple_value=float(value_loss))
            summary.value.add(tag='Policy Loss', simple_value=float(policy_loss))
            summary.value.add(tag='Var Norm Loss', simple_value=float(variable_norms))
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

        action_policy, value = self.session.run([self.localNetwork.actionPolicy, self.localNetwork.value],
                                                feed_dict={
                                                    self.localNetwork.screen: screen,
                                                    self.localNetwork.actionInfo: action_info,
                                                    self.localNetwork.generalFeatures: gen_features,
                                                    self.localNetwork.buildQueue: b_queue,
                                                    self.localNetwork.selection: selection,
                                                })

        # Select action from policies
        action, action_exp, spatial_action = self.select_action(action_policy,
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
        return [screen, action_exp, reward, value[0], spatial_action, gen_features, b_queue, selection], done, \
            screen, action_info, obs,

    def select_action(self, action_policy, obs, screen, genFeatures, bQueue, selection):

        # Find action

        # returns list of chosen action intersected with pysc available actions (currently available actions)
        vActions = getAvailableActions(obs, self.mapName)
        # flatten
        action_policy = np.ravel(action_policy)
        # Cuts off any unavailable actions
        valid_actions = action_policy[vActions]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabillities)
        normalized_actions = [float(i) / sum(valid_actions) for i in valid_actions]
        # Pick an action with probabillity normActions(gets original probability from
        action_prob = np.random.choice(len(vActions), p=normalized_actions)
        # validActions, not the normalized version) gives us action exploration
        if np.random.rand() < self.exploration:
            act_id = vActions[action_prob]
        else:
            act_id = vActions[np.argmax(action_policy[vActions])]

        spatial_policy = self.session.run([self.localNetwork.spatialPolicy],
                                          feed_dict={
                                             self.localNetwork.screen: screen,
                                             self.localNetwork.generalFeatures: genFeatures,
                                             self.localNetwork.buildQueue: bQueue,
                                             self.localNetwork.selection: selection
                                             })
        # Find spatial action
        spatial_action = np.ravel(spatial_policy)  # flatten
        spaction = np.random.choice((self.config.screen_size ** 2), 1, p=spatial_action)
        if np.random.rand() < self.exploration:
            spatial_action = [1, spaction]
        else:
            spatial_action = [1, np.argmax(spatial_action)]
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
        return [sc_actions.FunctionCall(act_id, act_args)], action_exp, spatial_action
