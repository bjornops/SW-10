import tensorflow as tf
import threading
import os
from time import sleep
from utils.config import process_config
from utils.dirs import create_dirs

from data_loader.data_generator import DataGenerator
from models.strategic_network import StrategicNetwork
from trainers.ctrl_random_options import StrategicRandOpt
from utils.logger import Logger
from absl import app
from utils.utils import load_strategic, tactical_network_setup


def main(argv):
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        config = process_config("../configs/test_config.json")

    except:
        print("missing or invalid config file")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # create instance of the model you want
    global_network_reference = StrategicNetwork(config, "global")  # exists in the TF session
    models = []
    for i in range(config.worker_count):
        model = StrategicNetwork(config, "worker_" + str(i) + "_scope")
        model.init_saver()
        model.init_worker_calc_variables()
        models.append(model)

    # create your data generator
    data = DataGenerator(config)  # TODO Remove?

    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainers = []
    for i in range(config.worker_count):
        trainer = StrategicRandOpt("worker_" + str(i), config, sess, models[i], data, logger)
        trainers.append(trainer)

    # Loading
    if config.load_model:
        sess.run(tf.global_variables_initializer())
        dict_strategic = load_strategic(config)
        saver_strategic = tf.train.Saver(var_list=dict_strategic, max_to_keep=5)
        pretrained_path = os.path.join(config.pretrained_dir, "strategic")  # "G:/pysc/models/BuildMarinesTBlue2"
        checkpoint = tf.train.get_checkpoint_state(pretrained_path)
        saver_strategic.restore(sess, checkpoint.model_checkpoint_path)
        print("Loading Strategic Model/Network")
    else:
        sess.run(tf.global_variables_initializer())

    # Tactical networks
    tactical_networks, dict_tacticals = tactical_network_setup(config)

    saver = []
    for i in range(5):
        saver.append(tf.train.Saver(var_list=dict_tacticals[i], max_to_keep=5))

    expArmyPath = os.path.join(config.pretrained_dir, "HHExpandArmy2")
    checkpoint = tf.train.get_checkpoint_state(expArmyPath) #Gets last model checkpoint
    saver[0].restore(sess, checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

    expProdPath = os.path.join(config.pretrained_dir, "HHBuildBarracks") # Expand Production
    checkpoint = tf.train.get_checkpoint_state(expProdPath) #Gets last model checkpoint
    saver[1].restore(sess, checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

    assignSCVPath = os.path.join(config.pretrained_dir, "HHAssignSCV")
    checkpoint = tf.train.get_checkpoint_state(assignSCVPath) #Gets last model checkpoint
    saver[2].restore(sess, checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

    supplyPath = os.path.join(config.pretrained_dir, "HHBuildSupply")
    checkpoint = tf.train.get_checkpoint_state(supplyPath) #Gets last model checkpoint
    saver[3].restore(sess, checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

    buildSCVPath = os.path.join(config.pretrained_dir, "HHBuildSCV")
    checkpoint = tf.train.get_checkpoint_state(buildSCVPath) #Gets last model checkpoint
    saver[4].restore(sess, checkpoint.model_checkpoint_path) #sets current network to that of checkpoint

    # here you train your model
    worker_handler(sess, trainers, config, tactical_networks)


def worker_handler(sess, trainers, config, tactical_networks):
    workers = []

    # Create workers
    for i in range(config.worker_count):
        workers.append(trainers[i])
        # keep n last saved models

    # Initialize global variables
    #sess.run(tf.global_variables_initializer())

    # tf class for simple coordinating of threads
    thread_coordinator = tf.train.Coordinator()

    # Start the work function of each worker in separate threads.
    worker_threads = []
    for worker in workers:
        # sets up thread with args for function
        worker_thread = threading.Thread(target=worker.work, args=(thread_coordinator, tactical_networks))
        worker_thread.start()
        sleep(0.5)
        worker_threads.append(worker_thread)

    # wait for all threads to finish
    thread_coordinator.join(worker_threads)


if __name__ == '__main__':
    app.run(main)
