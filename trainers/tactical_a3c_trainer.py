from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import tensorflow as tf
from base.base_train import BaseTrain
from models.tactical_network import TacticalNetwork
from pysc2.env import sc2_env
from pysc2.lib import actions as scActions

from utils.sw9_utilities import updateNetwork, addFeatureLayers, getAvailableActions, addGeneralFeatures


class TacticalTrainer(BaseTrain):
    def __init__(self, workerID, config, globalEpisodes, sess, model, data, logger):
        super(TacticalTrainer, self).__init__(sess, model, data, config, logger)
        self.config = config

        self.name = workerID
        self.globalEpisodes = globalEpisodes
        self.increment = self.globalEpisodes.assign_add(1)
        self.episodeRewards = []
        self.episodeMeans = []
        self.session = sess
        self.number_of_actions = len(scActions.FUNCTIONS)

        # Tensorflow summary writer (for tensorboard)
        # self.summaryWriter = tf.summary.FileWriter("log/train" + mapName + testID + "-" + str(self.number))
        self.screenSize = config.screen_size
        self.exploration = config.exploration
        self.mapName = config.map_name
        self.previousAction = np.zeros([1, 1], dtype=np.float32)

        # Create local network
        self.localNetwork = TacticalNetwork(self.config, self.name)

        # define that when running a tf session with self.updatevars we want to update the worker to the global network
        self.updateVars = updateNetwork('global', self.name)

        # Setup sc environment
        game = sc2_env.SC2Env(
            map_name=config.map_name,
            step_mul=config.step_mul,
            visualize=False)

        self.env = game

    def train(self, experienceBuffer, val, lr):
        experienceBuffer = np.array(experienceBuffer)

        # Splitting of experience
        observations = experienceBuffer[:, 0]
        actions = experienceBuffer[:, 1]
        rewards = experienceBuffer[:, 2]
        values = experienceBuffer[:, 3]
        spatialAction = experienceBuffer[:, 4]
        genFeatures = experienceBuffer[:, 5]
        obsBuild = experienceBuffer[:, 6]
        selections = experienceBuffer[:, 7]
        prevActions = experienceBuffer[:, 8]
        # stores available actions for each timestep
        actionInfos = []

        # Spatial and non spatial action preperation
        bufferSize = len(experienceBuffer)

        # used for storing whether the current action made use of a spatial action
        validSpatialAction = np.zeros([bufferSize], dtype=np.float32)
        # stores the picked spatial action(i[1]) for each experience tuple
        selectedSpatialAction = np.zeros([bufferSize, self.screenSize ** 2], dtype=np.float32)
        # stores which actions were valid at a given time
        validActions = np.zeros([bufferSize, self.number_of_actions], dtype=np.float32)
        # stores which action was selected at a given time
        selectedAction = np.zeros([bufferSize, self.number_of_actions], dtype=np.float32)
        # stores which action was selected at a given time
        actionMem = np.zeros([bufferSize, 1], dtype=np.float32)

        valueTarget = np.zeros([bufferSize], dtype=np.float32)

        if math.isnan(val):
            valueTarget[-1] = 0
        else:
            valueTarget[-1] = val

        # goes through each timestep in experience buffer
        for t in range(0, bufferSize):
            # was a spatial action used during this timestep
            if spatialAction[t][0] == 1:
                # set spatial action chosen during current timestep
                selectedSpatialAction[t, spatialAction[t][1]] = 1
                # a spatial action was used during this timestep
                validSpatialAction[t] = 1

            # set which actions were valid during this timestep
            validActions[t, actions[t][1]] = 1
            # set action chosen during this timestep
            selectedAction[t, actions[t][0]] = 1
            # stores valid actions for a specific timestep
            actionInfo = np.zeros([1, self.number_of_actions], dtype=np.float32)
            # sets valid actions for current timestep
            actionInfo[0, actions[t][1]] = 1
            actionInfos.append(actionInfo)
            actionMem[t] = prevActions[t]

        for t in range(bufferSize - 2, -1, -1):
            valueTarget[t] = rewards[t] + self.config.gamma * valueTarget[t + 1]

        # changes dimensions from [buffersize, 1, NB_actions] to [buffersize, NB_actions]
        actionInfos = np.concatenate(actionInfos, axis=0)

        # Define feed to use for Updating the global network using gradients from loss
        feed_dict = {self.localNetwork.valueTarget: valueTarget,
                     self.localNetwork.screen: np.vstack(observations),
                     self.localNetwork.actionInfo: actionInfos,
                     self.localNetwork.validActions: validActions,
                     self.localNetwork.selectedAction: selectedAction,
                     self.localNetwork.selectedSpatialAction: selectedSpatialAction,
                     self.localNetwork.validSpatialAction: validSpatialAction,
                     self.localNetwork.learningRate: lr,
                     self.localNetwork.generalFeatures: np.vstack(genFeatures),
                     self.localNetwork.buildQueue: np.vstack(obsBuild),
                     self.localNetwork.selection: np.vstack(selections),
                     self.localNetwork.previousActions: actionMem}

        # Generate statistics from our network to periodically save and start the network feed
        valueLoss, policyLoss, gradientNorms, variableNorms, _ = self.session.run([self.localNetwork.valueLoss,
                                                                              self.localNetwork.policyLoss,
                                                                              self.localNetwork.gradNorms,
                                                                              self.localNetwork.varNorms,
                                                                              self.localNetwork.applyGrads],
                                                                             feed_dict=feed_dict)
        # Returns statistics for our summary writer
        return valueLoss / len(experienceBuffer), policyLoss / len(experienceBuffer), gradientNorms, variableNorms

    def work(self, threadCoordinator):
        self.episodeCount = self.session.run(self.globalEpisodes)  # gets current global episode
        print("Starting worker '" + str(self.name) + "'")

        with self.session.as_default(), self.session.graph.as_default():
            while not threadCoordinator.should_stop():
                self.train_epoch()


    # Each episode
    def train_epoch(self):

        # reset local network to global network (updateVars = vars of global network)
        self.session.run(self.updateVars)
        # store experience tuples
        experienceBuffer = []
        # Store values
        episodeValues = []
        # Store Rewards
        episodeReward = 0
        # Is the minigame over?
        done = False

        # Reset minigame
        obs = self.env.reset()
        self.previousAction = np.zeros([1, 1], dtype=np.float32)

        # each step
        while not done:
            # perform step, return exp
            exp, done, screen, actionInfo, obs = self.train_step(obs)
            experienceBuffer.append(exp)

            episodeValues.append(exp[3])
            episodeReward += exp[2]

            if len(experienceBuffer) == self.config.buffer_size and not done:
                # we dont know what our final return is, so we bootstrap from our current value estimation.
                val = self.session.run(self.localNetwork.value,
                                       feed_dict={self.localNetwork.screen: screen,
                                                  self.localNetwork.actionInfo: actionInfo,
                                                  self.localNetwork.generalFeatures: experienceBuffer[-1][5],
                                                  self.localNetwork.buildQueue: experienceBuffer[-1][6],
                                                  self.localNetwork.selection: experienceBuffer[-1][7],
                                                  })[
                    0]  # self.localNetwork.previousActions: self.previousAction})[0]
                lr = self.config.learning_rate * (1 - 0.5 * self.episodeCount / self.config.total_episodes)
                valueLoss, policyLoss, gradientNorms, variableNorms = self.train(experienceBuffer, self.config.gamma,
                                                                                 val, lr)
                experienceBuffer = []
                self.session.run(self.updateVars)
            if done:
                break

        # When done == true
        self.episodeRewards.append(episodeReward)
        self.episodeMeans.append(np.mean(episodeValues))
        print("Episode: " + str(self.episodeCount) + " Reward: " + str(episodeReward))

        # Update the network using the experience buffer at the end of the episode.
        if len(experienceBuffer) != 0:
            lr = self.config.learning_rate * (1 - 0.5 * self.episodeCount / self.config.total_episodes)
            valueLoss, policyLoss, gradientNorms, variableNorms = self.train(experienceBuffer,
                                                                             self.config.gamma, 0, lr)

        # save model and statistics.
        if self.episodeCount != 0:
            # makes sure only one of our workers saves the model
            if self.episodeCount % 25 == 0 and self.name == 'worker_0':
                globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                dict = {self.mapName + '/sconv1/weights:0': globalVars[0],
                        self.mapName + '/sconv1/biases:0': globalVars[1],
                        self.mapName + '/sconv2/weights:0': globalVars[2],
                        self.mapName + '/sconv2/biases:0': globalVars[3],
                        self.mapName + '/info_fc/weights:0': globalVars[4],
                        self.mapName + '/info_fc/biases:0': globalVars[5],
                        self.mapName + '/genFc/weights:0': globalVars[6],
                        self.mapName + '/genFc/biases:0': globalVars[7],
                        self.mapName + '/spPol/weights:0': globalVars[8],
                        self.mapName + '/spPol/biases:0': globalVars[9],
                        self.mapName + '/feat_fc/weights:0': globalVars[10],
                        self.mapName + '/feat_fc/biases:0': globalVars[11],
                        self.mapName + '/non_spatial_action/weights:0': globalVars[12],
                        self.mapName + '/non_spatial_action/biases:0': globalVars[13],
                        self.mapName + '/value/weights:0': globalVars[14],
                        self.mapName + '/value/biases:0': globalVars[15]}
                saver = tf.train.Saver(dict)
                saver.save(self.session, self.modelPath + '/model-' + str(self.episodeCount) + '.cptk')
                print("Model Saved..")

            meanReward = np.mean(self.episodeRewards[-1:])
            meanValue = np.mean(self.episodeMeans[-1:])
            summary = tf.Summary()
            summary.value.add(tag='Reward', simple_value=float(meanReward))
            summary.value.add(tag='Value', simple_value=float(meanValue))
            summary.value.add(tag='Value Loss', simple_value=float(valueLoss))
            summary.value.add(tag='Policy Loss', simple_value=float(policyLoss))
            summary.value.add(tag='Grad Norm Loss', simple_value=float(gradientNorms))
            summary.value.add(tag='Var Norm Loss', simple_value=float(variableNorms))
            self.summaryWriter.add_summary(summary, self.episodeCount)

            self.summaryWriter.flush()  # flushes to disk

        if self.name == 'worker_0':  # TODO Maybe fix later.
            self.session.run(self.increment)

        self.episodeCount += 1

    def train_step(self, obs):
        # add feature layers
        screen = addFeatureLayers(obs[0])

        # run session and get policies
        actionInfo = np.zeros([1, self.number_of_actions], dtype=np.float32)
        # list of available actions
        actionInfo[0, getAvailableActions(obs[0])] = 1

        genFeatures, bQueue, selection = addGeneralFeatures(obs[0])

        actionPolicy, value = self.session.run([self.localNetwork.actionPolicy, self.localNetwork.value],
                                          feed_dict={self.localNetwork.screen: screen,
                                                     self.localNetwork.actionInfo: actionInfo,
                                                     self.localNetwork.generalFeatures: genFeatures,
                                                     self.localNetwork.buildQueue: bQueue,
                                                     self.localNetwork.selection: selection,
                                                     self.localNetwork.previousActions: self.previousAction})

        # Select action from policies
        action, actionExp, spatialAction = self.selectAction(actionPolicy, obs[0], screen, genFeatures,
                                                                 bQueue, selection)

        obs = self.env.step(action)  # Perform action on environment

        # self.previousAction = np.insert(self.previousAction, 0, actionExp[0], axis=1)
        # self.previousAction = np.delete(self.previousAction,4, 1)
        actionMem = self.previousAction[:]

        self.previousAction[0][0] = 0

        # Gets reward from current step
        reward = obs[0].reward
        # Check if the minigame has finished
        done = obs[0].last()

        # return experience
        return [screen, actionExp, reward, value[0], spatialAction, genFeatures, bQueue, selection, actionMem], done, \
               screen, actionInfo, obs,


    def selectAction(self, actionPolicy, obs, screen, genFeatures, bQueue, selection):

        # Find action

        # returns list of chosen action intersected with pysc available actions (currently available actions)
        vActions = getAvailableActions(obs)
        # flatten
        actionPolicy = np.ravel(actionPolicy)
        # Cuts off any unavailable actions
        validActions = actionPolicy[vActions]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabillities)
        normActions = [float(i) / sum(validActions) for i in validActions]
        # Pick an action with probabillity normActions(gets original probability from
        actionProb = np.random.choice(len(vActions), p=normActions)
        # validActions, not the normalized version) gives us action exploration
        if np.random.rand() < self.exploration:
            act_id = vActions[actionProb]
        else:
            act_id = vActions[np.argmax(actionPolicy[vActions])]

        spatialPolicy = self.session.run([self.localNetwork.spatialPolicy],
                                    feed_dict={self.localNetwork.screen: screen,
                                               self.localNetwork.generalFeatures: genFeatures,
                                               self.localNetwork.buildQueue: bQueue,
                                               self.localNetwork.selection: selection,
                                               self.localNetwork.previousActions: self.previousAction})
        # Find spatial action
        spatialAction = np.ravel(spatialPolicy)  # flatten
        spaction = np.random.choice((64 * 64), 1, p=spatialAction)
        if np.random.rand() < self.exploration:
            spatialAction = [1, spaction]
        else:
            spatialAction = [1, np.argmax(spatialAction)]
        target = [int(spatialAction[1] // self.screenSize), int(spatialAction[1] % self.screenSize)]

        spatialAction[1] = target[0] * self.screenSize + target[1]

        # define second spatial action Todo find a suitable solution
        target2 = target[:]
        target2[0] = int(max(0, min(self.screenSize - 1, target[0] + 6)))
        target2[1] = int(max(0, min(self.screenSize - 1, target[1] + 6)))
        if act_id == scActions.FUNCTIONS.select_rect.id:
            target[0] = int(max(0, min(self.screenSize - 1, target[0] - 6)))
            target[1] = int(max(0, min(self.screenSize - 1, target[1] - 6)))

        # For experience
        actionExp = [act_id, vActions]

        act_args = []
        for arg in scActions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap'):
                act_args.append([target[1], target[0]])
            elif arg.name in ('screen2'):
                act_args.append([target2[1], target2[0]])
            elif arg.name in ('control_group_id'):
                act_args.append([4])
            else:
                # No spatial action was used
                spatialAction[0] = 0
                act_args.append([0])
        return [scActions.FunctionCall(act_id, act_args)], actionExp, spatialAction
