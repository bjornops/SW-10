import tensorflow as tf
import threading
from time import sleep
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from data_loader.data_generator import DataGenerator
from models.tactical_network import TacticalNetwork
# from trainers.example_trainer import ExampleTrainer
from trainers.tactical_a3c_trainer import TacticalTrainer
from utils.logger import Logger
from absl import app
import sys


def main(argv):
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        config = process_config("../configs/test_config.json")

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # create instance of the model you want
    global_network_reference = TacticalNetwork(config, "global")  # exists in the TF session
    model = TacticalNetwork(config, "worker_0_scope")
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = TacticalTrainer("worker_0", config, sess, model, data, logger)

    # here you train your model
    #trainer.train()
    worker_handler(sess, trainer, model, config)


def worker_handler(sess, trainer, model, config):
    data = 0
    logger = 0
    workers = []

    # Create workers
    for i in range(config.worker_count):
        workers.append(trainer)
        # keep n last saved models

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # tf class for simple coordinating of threads
    thread_coordinator = tf.train.Coordinator()

    # Start the work function of each worker in seperate threads.
    worker_threads = []
    for worker in workers:
        # sets up thread with args for function
        worker_thread = threading.Thread(target=worker.work, args=(thread_coordinator,))
        worker_thread.start()
        sleep(0.5)
        worker_threads.append(worker_thread)

    # wait for all threads to finish
    thread_coordinator.join(worker_threads)


if __name__ == '__main__':
    app.run(main)
