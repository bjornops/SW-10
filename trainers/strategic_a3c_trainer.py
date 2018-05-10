from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf
from base.base_train import BaseTrain
from pysc2.env import sc2_env
from pysc2.lib import actions as sc_actions
from utils.utilities import updateNetwork, addFeatureLayers, getAvailableActions, addGeneralFeatures, \
    getAvailableActionsStrat, getAvailableActionsEA, getAvailableActionsBB, getAvailableActionsASCV, \
    getAvailableActionsBS, getAvailableActionsBSCV
from utils.CSV import StoreAsCSV


class StrategicTrainer(BaseTrain):
    def __init__(self, worker_id, config, sess, model, data, logger):
        super(StrategicTrainer, self).__init__(sess, model, data, config, logger)
        self.config = config

        self.name = worker_id
        self.episodeRewards = []
        self.episodeMeans = []
        self.session = sess
        self.number_of_actions = 5  # len(sc_actions.FUNCTIONS)
        self.experience_buffer = []
        self.val = 0

        self.option_log_list = []

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

    def work(self, thread_coordinator, tactical_networks):
        self.tactical_networks = tactical_networks
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
            if self.episode_count % 25 == 0 and self.name == 'worker_0':
                self.localNetwork.save(self.session)
                StoreAsCSV(self.option_log_list)
                self.option_log_list = []

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

    def perform_env_action(self, obs):
        # add feature layers
        screen = addFeatureLayers(obs[0])

        # run session and get policies
        action_info = np.zeros([1, self.number_of_actions], dtype=np.float32)
        # list of available actions
        action_info[0, getAvailableActionsStrat(obs[0])] = 1

        gen_features, b_queue, selection = addGeneralFeatures(obs[0])

        action_policy, value = self.session.run([self.localNetwork.actionPolicy, self.localNetwork.value],
                                                feed_dict={
                                                    self.localNetwork.screen: screen,
                                                    self.localNetwork.actionInfo: action_info,
                                                    self.localNetwork.generalFeatures: gen_features,
                                                    self.localNetwork.buildQueue: b_queue,
                                                    self.localNetwork.selection: selection,
                                                })

        selected_tactical = self.select_tactical(action_policy, obs[0])

        print("option:" + str(selected_tactical))
        reward = 0
        done = False
        cur_step = 0

        while not done and cur_step < self.config.option_timeout:
            # Select action from policies
            action, action_exp, spatial_action = self.select_action(selected_tactical,
                                                                    obs[0],
                                                                    screen,
                                                                    gen_features,
                                                                    b_queue,
                                                                    selection)

            obs = self.env.step(action)  # Perform action on environment

            # Gets reward from current step
            reward += obs[0].reward
            # Check if the minigame has finished
            done = obs[0].last()
            cur_step += 1

        # Adv. log
        selected_option = [selected_tactical, cur_step, reward, self.episode_count]
        self.option_log_list.append(selected_option)

        # return experience
        return [screen, action_exp, reward, value[0], spatial_action, gen_features, b_queue, selection], done, \
               screen, action_info, obs

    def select_tactical(self, action_policy, obs):
        # Find action
        # returns list of chosen action intersected with pysc available actions (currently available actions)
        v_actions = getAvailableActionsStrat(obs)
        # flatten
        action_policy = np.ravel(action_policy)
        # Cuts off any unavailable actions
        valid_actions = action_policy[v_actions]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabilities)
        norm_actions = [float(i) / sum(valid_actions) for i in valid_actions]
        # Pick an action with probability norm_actions(gets original probability from
        action_prob = np.random.choice(len(v_actions), p=norm_actions)
        # valid_actions, not the normalized version) gives us action exploration
        strat_act_id = v_actions[action_prob]

        return strat_act_id

    def select_action(self, strat_act_id, obs, screen, gen_features, b_queue, selection):
        session = self.session

        action_info = np.zeros([1, len(sc_actions.FUNCTIONS)], dtype=np.float32)
        # list of available actions
        # action_info[0, getAvailableActions(obs, self.config.map_name)] = 1

        # Set local tactical networks
        tact_net = self.tactical_networks[0]
        tact_net1 = self.tactical_networks[1]
        tact_net2 = self.tactical_networks[2]
        tact_net3 = self.tactical_networks[3]
        tact_net4 = self.tactical_networks[4]

        # Select primitive action
        if strat_act_id == 0:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            v_actions_tact = getAvailableActionsEA(obs)
            action_info[0, getAvailableActions(obs, "HHExpandArmy2")] = 1
            tact_action_policy = session.run([tact_net.actionPolicy],
                                             feed_dict={tact_net.screen: screen,
                                                        tact_net.actionInfo: action_info,
                                                        tact_net.generalFeatures: gen_features,
                                                        tact_net.buildQueue: b_queue,
                                                        tact_net.selection: selection})
        elif strat_act_id == 1:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            v_actions_tact = getAvailableActionsBB(obs)
            action_info[0, getAvailableActions(obs, "HHBuildBarracks")] = 1
            tact_action_policy = session.run([tact_net1.actionPolicy],
                                             feed_dict={tact_net1.screen: screen,
                                                        tact_net1.actionInfo: action_info,
                                                        tact_net1.generalFeatures: gen_features,
                                                        tact_net1.buildQueue: b_queue,
                                                        tact_net1.selection: selection})

        elif strat_act_id == 2:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            v_actions_tact = getAvailableActionsASCV(obs)
            action_info[0, getAvailableActions(obs, "HHAssignSCV")] = 1
            tact_action_policy = session.run([tact_net2.actionPolicy],
                                             feed_dict={tact_net2.screen: screen,
                                                        tact_net2.actionInfo: action_info,
                                                        tact_net2.generalFeatures: gen_features,
                                                        tact_net2.buildQueue: b_queue,
                                                        tact_net2.selection: selection})

        elif strat_act_id == 3:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            v_actions_tact = getAvailableActionsBS(obs)
            action_info[0, getAvailableActions(obs, "HHBuildSupply")] = 1
            tact_action_policy = session.run([tact_net3.actionPolicy],
                                             feed_dict={tact_net3.screen: screen,
                                                        tact_net3.actionInfo: action_info,
                                                        tact_net3.generalFeatures: gen_features,
                                                        tact_net3.buildQueue: b_queue,
                                                        tact_net3.selection: selection})
        elif strat_act_id == 4:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            v_actions_tact = getAvailableActionsBSCV(obs)
            action_info[0, getAvailableActions(obs, "HHBuildSCV")] = 1
            tact_action_policy = session.run([tact_net4.actionPolicy],
                                             feed_dict={tact_net4.screen: screen,
                                                        tact_net4.actionInfo: action_info,
                                                        tact_net4.generalFeatures: gen_features,
                                                        tact_net4.buildQueue: b_queue,
                                                        tact_net4.selection: selection})

        # Find action

        # returns list of chosen action intersected with pysc available actions (currently available actions)
        # v_actions_tact = getAvailableActions(obs)

        # flatten
        action_policy_tact = np.ravel(tact_action_policy)
        # Cuts off any unavailable actions
        valid_actions_tact = action_policy_tact[v_actions_tact]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabillities)
        norm_actions_tact = [float(i) / sum(valid_actions_tact) for i in valid_actions_tact]
        # Pick an action with probabillity norm_actions(gets original probability from
        action_prob_tact = np.random.choice(len(v_actions_tact), p=norm_actions_tact)
        act_id = v_actions_tact[action_prob_tact]

        # Selected action input for coordinate selection
        selected_action_array = np.zeros((1, len(sc_actions.FUNCTIONS)), dtype=np.float32)
        selected_action_array[0][act_id] = 1

        # Check whether coordinates are necessary
        is_spatial_action = False
        for arg in sc_actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                is_spatial_action = True
                
        if(is_spatial_action):
            if strat_act_id == 0:
                spatial_policy = session.run([tact_net.spatialPolicy],
                                             feed_dict={tact_net.screen: screen,
                                                        tact_net.generalFeatures: gen_features,
                                                        tact_net.buildQueue: b_queue,
                                                        tact_net.selection: selection,
                                                        tact_net.selected_action: selected_action_array
                                                        })
            elif strat_act_id == 1:
                spatial_policy = session.run([tact_net1.spatialPolicy],
                                             feed_dict={tact_net1.screen: screen,
                                                        tact_net1.generalFeatures: gen_features,
                                                        tact_net1.buildQueue: b_queue,
                                                        tact_net1.selection: selection,
                                                        tact_net1.selected_action: selected_action_array})
            elif strat_act_id == 2:
                spatial_policy = session.run([tact_net2.spatialPolicy],
                                             feed_dict={tact_net2.screen: screen,
                                                        tact_net2.generalFeatures: gen_features,
                                                        tact_net2.buildQueue: b_queue,
                                                        tact_net2.selection: selection,
                                                        tact_net2.selected_action: selected_action_array})
            elif strat_act_id == 3:
                spatial_policy = session.run([tact_net3.spatialPolicy],
                                             feed_dict={tact_net3.screen: screen,
                                                        tact_net3.generalFeatures: gen_features,
                                                        tact_net3.buildQueue: b_queue,
                                                        tact_net3.selection: selection,
                                                        tact_net3.selected_action: selected_action_array})
            elif strat_act_id == 4:
                spatial_policy = session.run([tact_net4.spatialPolicy],
                                             feed_dict={tact_net4.screen: screen,
                                                        tact_net4.generalFeatures: gen_features,
                                                        tact_net4.buildQueue: b_queue,
                                                        tact_net4.selection: selection,
                                                        tact_net4.selected_action: selected_action_array})

            # Find spatial action

            # flatten
            spatial_action = np.ravel(spatial_policy)
            spaction = np.random.choice((64 * 64), 1, p=spatial_action)
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
        v_actions = getAvailableActionsStrat(obs)
        action_exp = [strat_act_id, v_actions]

        act_args = []
        for arg in sc_actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap'):
                act_args.append([target[1], target[0]])
            elif arg.name in ('screen2'):
                act_args.append([target2[1], target2[0]])
            else:
                # No spatial action was used
                spatial_action = [0]
                spatial_action[0] = 0
                act_args.append([0])
        return [sc_actions.FunctionCall(act_id, act_args)], action_exp, spatial_action
