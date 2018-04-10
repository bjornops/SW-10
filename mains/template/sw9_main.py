import os
import threading
from time import sleep

# import multiprocessing  #only used if we want workercount = cpu count
import tensorflow as tf
from absl import app
from models.sw9_tactical_network_model import TacticalNetwork
# from absl import flags # substitute for gflags something.
from pysc2.lib import actions as scActions

from models.template.sw9_strategic_network_model import StrategicNetwork
from trainers.template.sw9_worker import Worker


def _main(unused_argv):

    # todo Change these
    # total amount of episodes to go through
    totalEpisodes = 2000
    # discount factor
    gamma = .99
    # Exploration prob
    exploration = 0.4
    # multiprocessing.cpu_count(), use this if you want worker count = cpu count
    workerCount = 1
    # Number of feature layers to include, make sure this fits with utilities/addFeatureLayers
    numberOfFeatures = 7
    # learningRate
    learningRate = 0.0001
    #beta #2+
    valueFactor = 2
    # eta
    entropyFactor = 0.00
    maxBufferSize = 80

    #Map to train on
    mapName = "BuildMarines"
    testID = "Black"
    # Use hierarki or standard A3C
    hier = False
    # Train?
    toTrain = True
    # whether to load a previously trained model
    loadModel = False

    numberOfActions = len(scActions.FUNCTIONS)
    if hier:
        numberOfActions = 5  # TODO Maybe add to config

    # path to save model
    modelPath = "C:/pysc/models/" + mapName + testID
    # size of screen screenSize*ScreenSize
    screenSize = 64
    stepMul = 8

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    startWorkers(totalEpisodes, gamma, screenSize, numberOfActions, loadModel, modelPath, workerCount,
                 numberOfFeatures, exploration, learningRate, stepMul, valueFactor, entropyFactor,
                 maxBufferSize, hier, toTrain, mapName, testID)


def startWorkers(totalEpisodes, gamma, screenSize, numberOfActions, loadModel, modelPath, workerCount,
                 numberOfFeatures, exploration, learningRate, stepMul, valueFactor, entropyFactor, maxBufferSize,
                 hier, toTrain, mapName, testID):
    # Clears the default graph stack and resets the global default graph (cleanup)
    tf.reset_default_graph()

    # Use CPU:0 for setup work
    with tf.device("/cpu:0"):
        # global counter of episodes
        globalEpisodes = tf.Variable(0, dtype=tf.int32, name='globalEpisodes', trainable=False)

        if hier:
            # Generate global network
            globalNetwork = StrategicNetwork(screenSize, numberOfActions, 'global',
                                             numberOfFeatures, valueFactor, entropyFactor)
        else:
            # Generate global network
            globalNetwork = TacticalNetwork(screenSize, numberOfActions, 'global',
                                            numberOfFeatures, valueFactor, entropyFactor)

        tactNetwork = {}
        tactNetwork1 = {}
        tactNetwork2 = {}
        tactNetwork3 = {}
        tactNetwork4 = {}

        if loadModel:
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            if hier:
                dictMain = {mapName + '/sconv1/weights:0': vars[0],
                            mapName + '/sconv1/biases:0': vars[1],
                            mapName + '/sconv2/weights:0': vars[2],
                            mapName + '/sconv2/biases:0': vars[3],
                            mapName + '/info_fc/weights:0': vars[4],
                            mapName + '/info_fc/biases:0': vars[5],
                            mapName + '/genFc/weights:0': vars[6],
                            mapName + '/genFc/biases:0': vars[7],
                            mapName + '/feat_fc/weights:0': vars[8],
                            mapName + '/feat_fc/biases:0': vars[9],
                            mapName + '/non_spatial_action/weights:0': vars[10],
                            mapName + '/non_spatial_action/biases:0': vars[11],
                            mapName + '/value/weights:0': vars[12],
                            mapName + '/value/biases:0': vars[13]}
            else:
                dictMain = {mapName + '/sconv1/weights:0': vars[0],
                            mapName + '/sconv1/biases:0': vars[1],
                            mapName + '/sconv2/weights:0': vars[2],
                            mapName + '/sconv2/biases:0': vars[3],
                            mapName + '/info_fc/weights:0': vars[4],
                            mapName + '/info_fc/biases:0': vars[5],
                            mapName + '/genFc/weights:0': vars[6],
                            mapName + '/genFc/biases:0': vars[7],
                            mapName + '/spPol/weights:0': vars[8],
                            mapName + '/spPol/biases:0': vars[9],
                            mapName + '/feat_fc/weights:0': vars[10],
                            mapName + '/feat_fc/biases:0': vars[11],
                            mapName + '/non_spatial_action/weights:0': vars[12],
                            mapName + '/non_spatial_action/biases:0': vars[13],
                            mapName + '/value/weights:0': vars[14],
                            mapName + '/value/biases:0': vars[15]}

        if hier:
            tactNetwork = TacticalNetwork(screenSize, len(scActions.FUNCTIONS),"HHExpandArmy2", numberOfFeatures,
                                           valueFactor, entropyFactor) # Create local network
            tactNetwork1 = TacticalNetwork(screenSize, len(scActions.FUNCTIONS),"HHBuildBarracks", numberOfFeatures,
                                           valueFactor, entropyFactor) # Create local network
            tactNetwork2 = TacticalNetwork(screenSize, len(scActions.FUNCTIONS),"HHAssignSCV", numberOfFeatures,
                                           valueFactor, entropyFactor) # Create local network
            tactNetwork3 = TacticalNetwork(screenSize, len(scActions.FUNCTIONS),"HHBuildSupply", numberOfFeatures,
                                           valueFactor, entropyFactor) # Create local network
            tactNetwork4 = TacticalNetwork(screenSize, len(scActions.FUNCTIONS),"HHBuildSCV",  numberOfFeatures,
                                           valueFactor, entropyFactor) # Create local network
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHExpandArmy2')

            dict = {'HHExpandArmy2/sconv1/weights:0' : vars[0],
                    'HHExpandArmy2/sconv1/biases:0' : vars[1],
                    'HHExpandArmy2/sconv2/weights:0' : vars[2],
                    'HHExpandArmy2/sconv2/biases:0' : vars[3],
                    'HHExpandArmy2/info_fc/weights:0' : vars[4],
                    'HHExpandArmy2/info_fc/biases:0' : vars[5],
                    'HHExpandArmy2/genFc/weights:0' : vars[6],
                    'HHExpandArmy2/genFc/biases:0' : vars[7],
                    'HHExpandArmy2/spPol/weights:0' : vars[8],
                    'HHExpandArmy2/spPol/biases:0' : vars[9],
                    'HHExpandArmy2/feat_fc/weights:0' : vars[10],
                    'HHExpandArmy2/feat_fc/biases:0' : vars[11],
                    'HHExpandArmy2/non_spatial_action/weights:0' : vars[12],
                    'HHExpandArmy2/non_spatial_action/biases:0' : vars[13],
                    'HHExpandArmy2/value/weights:0' : vars[14],
                    'HHExpandArmy2/value/biases:0' : vars[15]}

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildBarracks')

            dict2 = {'HHBuildBarracks/sconv1/weights:0' : vars[0],
                     'HHBuildBarracks/sconv1/biases:0' : vars[1],
                     'HHBuildBarracks/sconv2/weights:0' : vars[2],
                     'HHBuildBarracks/sconv2/biases:0' : vars[3],
                     'HHBuildBarracks/info_fc/weights:0' : vars[4],
                     'HHBuildBarracks/info_fc/biases:0' : vars[5],
                     'HHBuildBarracks/genFc/weights:0' : vars[6],
                     'HHBuildBarracks/genFc/biases:0' : vars[7],
                     'HHBuildBarracks/spPol/weights:0' : vars[8],
                     'HHBuildBarracks/spPol/biases:0' : vars[9],
                     'HHBuildBarracks/feat_fc/weights:0' : vars[10],
                     'HHBuildBarracks/feat_fc/biases:0' : vars[11],
                     'HHBuildBarracks/non_spatial_action/weights:0' : vars[12],
                     'HHBuildBarracks/non_spatial_action/biases:0' : vars[13],
                     'HHBuildBarracks/value/weights:0' : vars[14],
                     'HHBuildBarracks/value/biases:0' : vars[15]}

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHAssignSCV')

            dict3 = {'HHAssignSCV/sconv1/weights:0' : vars[0],
                     'HHAssignSCV/sconv1/biases:0' : vars[1],
                     'HHAssignSCV/sconv2/weights:0' : vars[2],
                     'HHAssignSCV/sconv2/biases:0' : vars[3],
                     'HHAssignSCV/info_fc/weights:0' : vars[4],
                     'HHAssignSCV/info_fc/biases:0' : vars[5],
                     'HHAssignSCV/genFc/weights:0' : vars[6],
                     'HHAssignSCV/genFc/biases:0' : vars[7],
                     'HHAssignSCV/spPol/weights:0' : vars[8],
                     'HHAssignSCV/spPol/biases:0' : vars[9],
                     'HHAssignSCV/feat_fc/weights:0' : vars[10],
                     'HHAssignSCV/feat_fc/biases:0' : vars[11],
                     'HHAssignSCV/non_spatial_action/weights:0' : vars[12],
                     'HHAssignSCV/non_spatial_action/biases:0' : vars[13],
                     'HHAssignSCV/value/weights:0' : vars[14],
                     'HHAssignSCV/value/biases:0' : vars[15]}

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildSupply')

            dict4 = {'HHBuildSupply/sconv1/weights:0' : vars[0],
                     'HHBuildSupply/sconv1/biases:0' : vars[1],
                     'HHBuildSupply/sconv2/weights:0' : vars[2],
                     'HHBuildSupply/sconv2/biases:0' : vars[3],
                     'HHBuildSupply/info_fc/weights:0' : vars[4],
                     'HHBuildSupply/info_fc/biases:0' : vars[5],
                     'HHBuildSupply/genFc/weights:0' : vars[6],
                     'HHBuildSupply/genFc/biases:0' : vars[7],
                     'HHBuildSupply/spPol/weights:0' : vars[8],
                     'HHBuildSupply/spPol/biases:0' : vars[9],
                     'HHBuildSupply/feat_fc/weights:0' : vars[10],
                     'HHBuildSupply/feat_fc/biases:0' : vars[11],
                     'HHBuildSupply/non_spatial_action/weights:0' : vars[12],
                     'HHBuildSupply/non_spatial_action/biases:0' : vars[13],
                     'HHBuildSupply/value/weights:0' : vars[14],
                     'HHBuildSupply/value/biases:0' : vars[15]}

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildSCV')

            dict5 = {'HHBuildSCV/sconv1/weights:0' : vars[0],
                     'HHBuildSCV/sconv1/biases:0' : vars[1],
                     'HHBuildSCV/sconv2/weights:0' : vars[2],
                     'HHBuildSCV/sconv2/biases:0' : vars[3],
                     'HHBuildSCV/info_fc/weights:0' : vars[4],
                     'HHBuildSCV/info_fc/biases:0' : vars[5],
                     'HHBuildSCV/genFc/weights:0' : vars[6],
                     'HHBuildSCV/genFc/biases:0' : vars[7],
                     'HHBuildSCV/spPol/weights:0' : vars[8],
                     'HHBuildSCV/spPol/biases:0' : vars[9],
                     'HHBuildSCV/feat_fc/weights:0' : vars[10],
                     'HHBuildSCV/feat_fc/biases:0' : vars[11],
                     'HHBuildSCV/non_spatial_action/weights:0' : vars[12],
                     'HHBuildSCV/non_spatial_action/biases:0' : vars[13],
                     'HHBuildSCV/value/weights:0' : vars[14],
                     'HHBuildSCV/value/biases:0' : vars[15]}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:

        if hier:
            saver1 = tf.train.Saver(var_list=dict, max_to_keep=10)
            saver2 = tf.train.Saver(var_list=dict2, max_to_keep=10)
            saver3 = tf.train.Saver(var_list=dict3, max_to_keep=10)
            saver4 = tf.train.Saver(var_list=dict4, max_to_keep=10)
            saver5 = tf.train.Saver(var_list=dict5, max_to_keep=10)

        if loadModel:
            saverMain = tf.train.Saver(var_list=dictMain,max_to_keep=10)



        workers = []
        # Create workers
        for i in range(workerCount):
            workers.append(Worker(i, screenSize, numberOfActions, modelPath, globalEpisodes, numberOfFeatures,
                                  exploration, stepMul, valueFactor, entropyFactor, learningRate, hier, toTrain,
                                  mapName, testID))
        # keep n last saved models
        saver = tf.train.Saver(max_to_keep=10)
        # Initialize global variables
        session.run(tf.global_variables_initializer())
        if loadModel:
            modPath = "G:/pysc/models/BuildMarinesTBlue2"
            # Gets last model checkpoint
            checkpoint = tf.train.get_checkpoint_state(modPath)
            # sets current network to that of checkpoint
            saverMain.restore(session, checkpoint.model_checkpoint_path)
        else:
            # Initialize global variables
            session.run(tf.global_variables_initializer())

        if hier:
            expArmyPath = "G:/pysc/models/HHExpandArmy2Blue5"
            checkpoint = tf.train.get_checkpoint_state(expArmyPath) #Gets last model checkpoint
            saver1.restore(session,checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

            expProdPath = "G:/pysc/models/HHBuildBarracksBlack2"
            checkpoint = tf.train.get_checkpoint_state(expProdPath) #Gets last model checkpoint
            saver2.restore(session,checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

            assignSCVPath = "G:/pysc/models/HHAssignSCVTeal8"
            checkpoint = tf.train.get_checkpoint_state(assignSCVPath) #Gets last model checkpoint
            saver3.restore(session,checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

            supplyPath = "G:/pysc/models/HHBuildSupplyBlack"
            checkpoint = tf.train.get_checkpoint_state(supplyPath) #Gets last model checkpoint
            saver4.restore(session,checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

            buildSCVPath = "G:/pysc/models/HHBuildSCVBlack2"
            checkpoint = tf.train.get_checkpoint_state(buildSCVPath) #Gets last model checkpoint
            saver5.restore(session,checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

        # tf class for simple coordinating of threads
        threadCoordinator = tf.train.Coordinator()

        # Start the work function of each worker in seperate threads.
        workerThreads = []
        for worker in workers:
            #sets up thread with args for function
            workerThread = threading.Thread(target=worker.work, args = (totalEpisodes, gamma, session,
                                                                        threadCoordinator, saver, numberOfActions,
                                                                        maxBufferSize, learningRate, tactNetwork,
                                                                        tactNetwork1, tactNetwork2, tactNetwork3,
                                                                        tactNetwork4))
            workerThread.start()
            sleep(0.5)
            workerThreads.append(workerThread)

        # wait for all threads to finish
        threadCoordinator.join(workerThreads)

if __name__ == "__main__":
    app.run(_main)