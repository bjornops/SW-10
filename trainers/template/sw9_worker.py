from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
from models.sw9_tactical_network_model import TacticalNetwork
from pysc2.env import sc2_env
from pysc2.lib import actions as scActions

from models.template.sw9_strategic_network_model import StrategicNetwork
from utils.sw9_utilities import updateNetwork, addFeatureLayers, getAvailableActions, addGeneralFeatures, \
    getAvailableActionsStrat, getAvailableActionsCM, getAvailableActionsEP, getAvailableActionsEA, \
    getAvailableActionsBS, getAvailableActionsBSCV  # , addFeatureLayersEA


class Worker():
    def __init__(self, workerID, screenSize, numberOfActions, modelPath, globalEpisodes, numberOfFeatures, exploration,
                 stepMul, valueFactor, entropyFactor, learningRate, hier, toTrain, mapName, testID):

        self.name = "worker_" + str(workerID)
        self.number = workerID
        self.modelPath = modelPath
        self.globalEpisodes = globalEpisodes
        self.increment = self.globalEpisodes.assign_add(1)
        self.episodeRewards = []
        self.episodeMeans = []
        # Tensorflow summary writer (for tensorboard)
        self.summaryWriter = tf.summary.FileWriter("log/train" + mapName + testID + "-" + str(self.number))
        self.screenSize = screenSize
        self.exploration = exploration
        self.toTrain = toTrain
        self.hier = hier
        self.mapName = mapName
        self.previousAction = np.zeros([1, 1], dtype=np.float32)
        if hier:
            # Create local network
            self.localNetwork = StrategicNetwork(screenSize, numberOfActions, self.name, numberOfFeatures, valueFactor,
                                                 entropyFactor)
        else:
            # Create local network
            self.localNetwork = TacticalNetwork(screenSize, numberOfActions, self.name, numberOfFeatures, valueFactor,
                                                entropyFactor)

        # define that when running a tf session with self.updatevars we want to update the worker to the global network
        self.updateVars = updateNetwork('global', self.name)

        # Setup sc environment
        game = sc2_env.SC2Env(
            map_name=mapName,
            step_mul=stepMul,
            visualize=False)

        self.env = game

    def train(self, experienceBuffer, session, gamma, numberOfActions, val, lr):
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
        validActions = np.zeros([bufferSize, numberOfActions], dtype=np.float32)
        # stores which action was selected at a given time
        selectedAction = np.zeros([bufferSize, numberOfActions], dtype=np.float32)
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
            actionInfo = np.zeros([1, numberOfActions], dtype=np.float32)
            # sets valid actions for current timestep
            actionInfo[0, actions[t][1]] = 1
            actionInfos.append(actionInfo)
            actionMem[t] = prevActions[t]

        for t in range(bufferSize - 2, -1, -1):
            valueTarget[t] = rewards[t] + gamma * valueTarget[t + 1]

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
        valueLoss, policyLoss, gradientNorms, variableNorms, _ = session.run([self.localNetwork.valueLoss,
                                                                              self.localNetwork.policyLoss,
                                                                              self.localNetwork.gradNorms,
                                                                              self.localNetwork.varNorms,
                                                                              self.localNetwork.applyGrads],
                                                                             feed_dict=feed_dict)
        # Returns statistics for our summary writer
        return valueLoss / len(experienceBuffer), policyLoss / len(experienceBuffer), gradientNorms, variableNorms

    def work(self, totalEpisodes, gamma, session, threadCoordinator, saver, numberOfActions,
             maxBufferSize, learningRate, tactNet, tactNet1, tactNet2, tactNet3, tactNet4):
        episodeCount = session.run(self.globalEpisodes)  # gets current global episode
        print("Starting worker " + str(self.number))

        with session.as_default(), session.graph.as_default():
            # Each episode
            while not threadCoordinator.should_stop():
                # reset local network to global network (updateVars = vars of global network)
                session.run(self.updateVars)
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
                    exp, done, screen, actionInfo, obs = self.step(session, numberOfActions, obs, tactNet,
                                                                   tactNet1, tactNet2, tactNet3, tactNet4)
                    experienceBuffer.append(exp)

                    episodeValues.append(exp[3])
                    episodeReward += exp[2]

                    if len(experienceBuffer) == maxBufferSize and not done and self.toTrain:
                        # we dont know what our final return is, so we bootstrap from our current value estimation.
                        val = session.run(self.localNetwork.value,
                                          feed_dict={self.localNetwork.screen: screen,
                                                     self.localNetwork.actionInfo: actionInfo,
                                                     self.localNetwork.generalFeatures: experienceBuffer[-1][5],
                                                     self.localNetwork.buildQueue: experienceBuffer[-1][6],
                                                     self.localNetwork.selection: experienceBuffer[-1][7],
                                                     })[
                            0]  # self.localNetwork.previousActions: self.previousAction})[0]
                        lr = learningRate * (1 - 0.5 * episodeCount / totalEpisodes)
                        valueLoss, policyLoss, gradientNorms, variableNorms = self.train(experienceBuffer,
                                                                                         session, gamma,
                                                                                         numberOfActions,
                                                                                         val, lr)
                        experienceBuffer = []
                        session.run(self.updateVars)
                    if done:
                        break

                # When done == true
                self.episodeRewards.append(episodeReward)
                self.episodeMeans.append(np.mean(episodeValues))
                print("Episode: " + str(episodeCount) + " Reward: " + str(episodeReward))
                # Update the network using the experience buffer at the end of the episode.
                if len(experienceBuffer) != 0 and self.toTrain:
                    lr = learningRate * (1 - 0.5 * episodeCount / totalEpisodes)
                    valueLoss, policyLoss, gradientNorms, variableNorms = self.train(experienceBuffer, session,
                                                                                     gamma, numberOfActions, 0, lr)

                # save model and statistics.
                if episodeCount % 1 == 0 and episodeCount != 0:
                    # makes sure only one of our workers saves the model
                    if episodeCount % 25 == 0 and self.name == 'worker_0':
                        globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                        if self.hier:
                            dict = {self.mapName + '/sconv1/weights:0': globalVars[0],
                                    self.mapName + '/sconv1/biases:0': globalVars[1],
                                    self.mapName + '/sconv2/weights:0': globalVars[2],
                                    self.mapName + '/sconv2/biases:0': globalVars[3],
                                    self.mapName + '/info_fc/weights:0': globalVars[4],
                                    self.mapName + '/info_fc/biases:0': globalVars[5],
                                    self.mapName + '/genFc/weights:0': globalVars[6],
                                    self.mapName + '/genFc/biases:0': globalVars[7],
                                    self.mapName + '/feat_fc/weights:0': globalVars[8],
                                    self.mapName + '/feat_fc/biases:0': globalVars[9],
                                    self.mapName + '/non_spatial_action/weights:0': globalVars[10],
                                    self.mapName + '/non_spatial_action/biases:0': globalVars[11],
                                    self.mapName + '/value/weights:0': globalVars[12],
                                    self.mapName + '/value/biases:0': globalVars[13]}
                        else:
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
                        saver.save(session, self.modelPath + '/model-' + str(episodeCount) + '.cptk')
                        print("Model Saved..")

                    meanReward = np.mean(self.episodeRewards[-1:])
                    meanValue = np.mean(self.episodeMeans[-1:])
                    summary = tf.Summary()
                    summary.value.add(tag='Reward', simple_value=float(meanReward))
                    if self.toTrain:
                        summary.value.add(tag='Value', simple_value=float(meanValue))
                        summary.value.add(tag='Value Loss', simple_value=float(valueLoss))
                        summary.value.add(tag='Policy Loss', simple_value=float(policyLoss))
                        summary.value.add(tag='Grad Norm Loss', simple_value=float(gradientNorms))
                        summary.value.add(tag='Var Norm Loss', simple_value=float(variableNorms))
                    self.summaryWriter.add_summary(summary, episodeCount)

                    self.summaryWriter.flush()  # flushes to disk
                if self.name == 'worker_0':
                    session.run(self.increment)
                episodeCount += 1

    def step(self, session, numberOfActions, obs, tactNet, tactNet1, tactNet2, tactNet3, tactNet4):
        # add feature layers
        screen = addFeatureLayers(obs[0])

        # run session and get policies
        actionInfo = np.zeros([1, numberOfActions], dtype=np.float32)
        if self.hier:
            # list of available actions
            actionInfo[0, getAvailableActionsStrat(obs[0])] = 1
        else:
            # list of available actions
            actionInfo[0, getAvailableActions(obs[0])] = 1

        genFeatures, bQueue, selection = addGeneralFeatures(obs[0])

        if self.hier:
            actionPolicy, value = session.run([self.localNetwork.actionPolicy, self.localNetwork.value],
                                              feed_dict={self.localNetwork.screen: screen,
                                                         self.localNetwork.actionInfo: actionInfo,
                                                         self.localNetwork.generalFeatures: genFeatures,
                                                         self.localNetwork.buildQueue: bQueue,
                                                         self.localNetwork.selection: selection})
        else:
            actionPolicy, value = session.run([self.localNetwork.actionPolicy, self.localNetwork.value],
                                              feed_dict={self.localNetwork.screen: screen,
                                                         self.localNetwork.actionInfo: actionInfo,
                                                         self.localNetwork.generalFeatures: genFeatures,
                                                         self.localNetwork.buildQueue: bQueue,
                                                         self.localNetwork.selection: selection,
                                                         self.localNetwork.previousActions: self.previousAction})

        # Select action from policies
        if self.hier:
            action, actionExp, spatialAction, actID = self.selectActionHier(actionPolicy, obs[0], screen, session,
                                                                            genFeatures, bQueue, tactNet, tactNet1,
                                                                            tactNet2, selection, tactNet3, tactNet4)
        else:
            action, actionExp, spatialAction = self.selectAction(actionPolicy, obs[0], screen, session, genFeatures,
                                                                 bQueue, selection)

        obs = self.env.step(action)  # Perform action on environment

        # self.previousAction = np.insert(self.previousAction, 0, actionExp[0], axis=1)
        # self.previousAction = np.delete(self.previousAction,4, 1)
        actionMem = self.previousAction[:]

        if self.hier:
            if actID == scActions.FUNCTIONS.select_idle_worker.id \
                    or actID == scActions.FUNCTIONS.select_point.id \
                    or actID == scActions.FUNCTIONS.select_rect.id:
                self.previousAction[0][0] = 1
            else:
                self.previousAction[0][0] = 0

        # Gets reward from current step
        reward = obs[0].reward
        # Check if the minigame has finished
        done = obs[0].last()

        # return experience
        return [screen, actionExp, reward, value[0], spatialAction, genFeatures, bQueue, selection, actionMem], done, \
               screen, actionInfo, obs,

    def selectActionHier(self, actionPolicy, obs, screen, session, genFeatures, bQueue, tactNet, tactNet1, tactNet2,
                         selection, tactNet3, tactNet4):

        # Find action
        # returns list of chosen action intersected with pysc available actions (currently available actions)
        vActions = getAvailableActionsStrat(obs)
        # flatten
        actionPolicy = np.ravel(actionPolicy)
        # Cuts off any unavailable actions
        validActions = actionPolicy[vActions]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabilities)
        normActions = [float(i) / sum(validActions) for i in validActions]
        # Pick an action with probabillity normActions(gets original probability from
        actionProb = np.random.choice(len(vActions), p=normActions)
        # validActions, not the normalized version) gives us action exploration
        if np.random.rand() < (self.exploration / 3):
            stratAct_id = np.random.choice(len(vActions))
        elif np.random.rand() < self.exploration:
            stratAct_id = vActions[actionProb]
        else:
            stratAct_id = vActions[np.argmax(actionPolicy[vActions])]

        actionInfo = np.zeros([1, len(scActions.FUNCTIONS)], dtype=np.float32)
        # list of available actions
        actionInfo[0, getAvailableActions(obs)] = 1
        # print(str(stratAct_id))
        if stratAct_id == 0:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            vActionsTact = getAvailableActionsEA(obs)

            tactActionPolicy = session.run([tactNet.actionPolicy],
                                           feed_dict={tactNet.screen: screen,
                                                      tactNet.actionInfo: actionInfo,
                                                      tactNet.generalFeatures: genFeatures,
                                                      tactNet.buildQueue: bQueue,
                                                      tactNet.selection: selection,
                                                      tactNet.previousActions: self.previousAction})
        elif stratAct_id == 1:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            vActionsTact = getAvailableActionsEP(obs)
            tactActionPolicy = session.run([tactNet1.actionPolicy],
                                           feed_dict={tactNet1.screen: screen,
                                                      tactNet1.actionInfo: actionInfo,
                                                      tactNet1.generalFeatures: genFeatures,
                                                      tactNet1.buildQueue: bQueue,
                                                      tactNet1.selection: selection,
                                                      tactNet1.previousActions: self.previousAction})

        elif stratAct_id == 2:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            vActionsTact = getAvailableActionsCM(obs)
            tactActionPolicy = session.run([tactNet2.actionPolicy],
                                           feed_dict={tactNet2.screen: screen,
                                                      tactNet2.actionInfo: actionInfo,
                                                      tactNet2.generalFeatures: genFeatures,
                                                      tactNet2.buildQueue: bQueue,
                                                      tactNet2.selection: selection,
                                                      tactNet2.previousActions: self.previousAction})

        elif stratAct_id == 3:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            vActionsTact = getAvailableActionsBS(obs)
            tactActionPolicy = session.run([tactNet3.actionPolicy],
                                           feed_dict={tactNet3.screen: screen,
                                                      tactNet3.actionInfo: actionInfo,
                                                      tactNet3.generalFeatures: genFeatures,
                                                      tactNet3.buildQueue: bQueue,
                                                      tactNet3.selection: selection,
                                                      tactNet3.previousActions: self.previousAction})
        elif stratAct_id == 4:
            # returns list of chosen action intersected with pysc available actions (currently available actions)
            vActionsTact = getAvailableActionsBSCV(obs)
            tactActionPolicy = session.run([tactNet4.actionPolicy],
                                           feed_dict={tactNet4.screen: screen,
                                                      tactNet4.actionInfo: actionInfo,
                                                      tactNet4.generalFeatures: genFeatures,
                                                      tactNet4.buildQueue: bQueue,
                                                      tactNet4.selection: selection,
                                                      tactNet4.previousActions: self.previousAction})

        # Find action

        # returns list of chosen action intersected with pysc available actions (currently available actions)
        # vActionsTact = getAvailableActions(obs)

        # flatten
        actionPolicyTact = np.ravel(tactActionPolicy)
        # Cuts off any unavailable actions
        validActionsTact = actionPolicyTact[vActionsTact]
        # Normalize the valid actions to get a probability distribution (since we cut away some/most probabillities)
        normActionsTact = [float(i) / sum(validActionsTact) for i in validActionsTact]
        # Pick an action with probabillity normActions(gets original probability from
        actionProbTact = np.random.choice(len(vActionsTact), p=normActionsTact)
        # validActions, not the normalized version) gives us action exploration
        if np.random.rand() < self.exploration:
            act_id = vActionsTact[actionProbTact]
        else:
            act_id = vActionsTact[np.argmax(actionPolicyTact[vActionsTact])]

        if stratAct_id == 0:
            spatialPolicy = session.run([tactNet.spatialPolicy],
                                        feed_dict={tactNet.screen: screen,
                                                   tactNet.generalFeatures: genFeatures,
                                                   tactNet.buildQueue: bQueue,
                                                   tactNet.selection: selection})
        elif stratAct_id == 1:
            spatialPolicy = session.run([tactNet1.spatialPolicy],
                                        feed_dict={tactNet1.screen: screen,
                                                   tactNet1.generalFeatures: genFeatures,
                                                   tactNet1.buildQueue: bQueue,
                                                   tactNet1.selection: selection})
        elif stratAct_id == 2:
            spatialPolicy = session.run([tactNet2.spatialPolicy],
                                        feed_dict={tactNet2.screen: screen,
                                                   tactNet2.generalFeatures: genFeatures,
                                                   tactNet2.buildQueue: bQueue,
                                                   tactNet2.selection: selection})
        elif stratAct_id == 3:
            spatialPolicy = session.run([tactNet3.spatialPolicy],
                                        feed_dict={tactNet3.screen: screen,
                                                   tactNet3.generalFeatures: genFeatures,
                                                   tactNet3.buildQueue: bQueue,
                                                   tactNet3.selection: selection})
        elif stratAct_id == 4:
            spatialPolicy = session.run([tactNet4.spatialPolicy],
                                        feed_dict={tactNet4.screen: screen,
                                                   tactNet4.generalFeatures: genFeatures,
                                                   tactNet4.buildQueue: bQueue,
                                                   tactNet4.selection: selection})

        # Find spatial action

        # flatten
        spatialAction = np.ravel(spatialPolicy)
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
        actionExp = [stratAct_id, vActions]

        act_args = []
        for arg in scActions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap'):
                act_args.append([target[1], target[0]])
            elif arg.name in ('screen2'):
                act_args.append([target2[1], target2[0]])
            else:
                # No spatial action was used
                spatialAction[0] = 0
                act_args.append([0])
        return [scActions.FunctionCall(act_id, act_args)], actionExp, spatialAction, act_id

    def selectAction(self, actionPolicy, obs, screen, session, genFeatures, bQueue, selection):

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

        spatialPolicy = session.run([self.localNetwork.spatialPolicy],
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
