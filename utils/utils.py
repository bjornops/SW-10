import argparse
from models.tactical_network import TacticalNetwork
import tensorflow as tf


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def tactical_network_setup(config):
    tactNetwork0 = TacticalNetwork(config, "HHExpandArmy2")  # Create local network
    tactNetwork1 = TacticalNetwork(config, "HHBuildBarracks")  # Create local network
    tactNetwork2 = TacticalNetwork(config, "HHAssignSCV")  # Create local network
    tactNetwork3 = TacticalNetwork(config, "HHBuildSupply")  # Create local network
    tactNetwork4 = TacticalNetwork(config, "HHBuildSCV")  # Create local network
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHExpandArmy2')

    dict1 = {'HHExpandArmy2/sconv1/weights:0': vars[0],
             'HHExpandArmy2/sconv1/biases:0': vars[1],
             'HHExpandArmy2/sconv2/weights:0': vars[2],
             'HHExpandArmy2/sconv2/biases:0': vars[3],
             'HHExpandArmy2/info_fc/weights:0': vars[4],
             'HHExpandArmy2/info_fc/biases:0': vars[5],
             'HHExpandArmy2/genFc/weights:0': vars[6],
             'HHExpandArmy2/genFc/biases:0': vars[7],
             'HHExpandArmy2/spPol/weights:0': vars[8],
             'HHExpandArmy2/spPol/biases:0': vars[9],
             'HHExpandArmy2/feat_fc/weights:0': vars[10],
             'HHExpandArmy2/feat_fc/biases:0': vars[11],
             'HHExpandArmy2/non_spatial_action/weights:0': vars[12],
             'HHExpandArmy2/non_spatial_action/biases:0': vars[13],
             'HHExpandArmy2/value/weights:0': vars[14],
             'HHExpandArmy2/value/biases:0': vars[15]}

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildBarracks')

    dict2 = {'HHBuildBarracks/sconv1/weights:0': vars[0],
             'HHBuildBarracks/sconv1/biases:0': vars[1],
             'HHBuildBarracks/sconv2/weights:0': vars[2],
             'HHBuildBarracks/sconv2/biases:0': vars[3],
             'HHBuildBarracks/info_fc/weights:0': vars[4],
             'HHBuildBarracks/info_fc/biases:0': vars[5],
             'HHBuildBarracks/genFc/weights:0': vars[6],
             'HHBuildBarracks/genFc/biases:0': vars[7],
             'HHBuildBarracks/spPol/weights:0': vars[8],
             'HHBuildBarracks/spPol/biases:0': vars[9],
             'HHBuildBarracks/feat_fc/weights:0': vars[10],
             'HHBuildBarracks/feat_fc/biases:0': vars[11],
             'HHBuildBarracks/non_spatial_action/weights:0': vars[12],
             'HHBuildBarracks/non_spatial_action/biases:0': vars[13],
             'HHBuildBarracks/value/weights:0': vars[14],
             'HHBuildBarracks/value/biases:0': vars[15]}

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHAssignSCV')

    dict3 = {'HHAssignSCV/sconv1/weights:0': vars[0],
             'HHAssignSCV/sconv1/biases:0': vars[1],
             'HHAssignSCV/sconv2/weights:0': vars[2],
             'HHAssignSCV/sconv2/biases:0': vars[3],
             'HHAssignSCV/info_fc/weights:0': vars[4],
             'HHAssignSCV/info_fc/biases:0': vars[5],
             'HHAssignSCV/genFc/weights:0': vars[6],
             'HHAssignSCV/genFc/biases:0': vars[7],
             'HHAssignSCV/spPol/weights:0': vars[8],
             'HHAssignSCV/spPol/biases:0': vars[9],
             'HHAssignSCV/feat_fc/weights:0': vars[10],
             'HHAssignSCV/feat_fc/biases:0': vars[11],
             'HHAssignSCV/non_spatial_action/weights:0': vars[12],
             'HHAssignSCV/non_spatial_action/biases:0': vars[13],
             'HHAssignSCV/value/weights:0': vars[14],
             'HHAssignSCV/value/biases:0': vars[15]}

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildSupply')

    dict4 = {'HHBuildSupply/sconv1/weights:0': vars[0],
             'HHBuildSupply/sconv1/biases:0': vars[1],
             'HHBuildSupply/sconv2/weights:0': vars[2],
             'HHBuildSupply/sconv2/biases:0': vars[3],
             'HHBuildSupply/info_fc/weights:0': vars[4],
             'HHBuildSupply/info_fc/biases:0': vars[5],
             'HHBuildSupply/genFc/weights:0': vars[6],
             'HHBuildSupply/genFc/biases:0': vars[7],
             'HHBuildSupply/spPol/weights:0': vars[8],
             'HHBuildSupply/spPol/biases:0': vars[9],
             'HHBuildSupply/feat_fc/weights:0': vars[10],
             'HHBuildSupply/feat_fc/biases:0': vars[11],
             'HHBuildSupply/non_spatial_action/weights:0': vars[12],
             'HHBuildSupply/non_spatial_action/biases:0': vars[13],
             'HHBuildSupply/value/weights:0': vars[14],
             'HHBuildSupply/value/biases:0': vars[15]}

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HHBuildSCV')

    dict5 = {'HHBuildSCV/sconv1/weights:0': vars[0],
             'HHBuildSCV/sconv1/biases:0': vars[1],
             'HHBuildSCV/sconv2/weights:0': vars[2],
             'HHBuildSCV/sconv2/biases:0': vars[3],
             'HHBuildSCV/info_fc/weights:0': vars[4],
             'HHBuildSCV/info_fc/biases:0': vars[5],
             'HHBuildSCV/genFc/weights:0': vars[6],
             'HHBuildSCV/genFc/biases:0': vars[7],
             'HHBuildSCV/spPol/weights:0': vars[8],
             'HHBuildSCV/spPol/biases:0': vars[9],
             'HHBuildSCV/feat_fc/weights:0': vars[10],
             'HHBuildSCV/feat_fc/biases:0': vars[11],
             'HHBuildSCV/non_spatial_action/weights:0': vars[12],
             'HHBuildSCV/non_spatial_action/biases:0': vars[13],
             'HHBuildSCV/value/weights:0': vars[14],
             'HHBuildSCV/value/biases:0': vars[15]}

    tactical_networks = [tactNetwork0, tactNetwork1, tactNetwork2, tactNetwork3, tactNetwork4]
    dict = [dict1, dict2, dict3, dict4, dict5]

    return tactical_networks, dict


def load_strategic(config):

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

    dict = {config.map_name + '/sconv1/weights:0': vars[0],
            config.map_name + '/sconv1/biases:0': vars[1],
            config.map_name + '/sconv2/weights:0': vars[2],
            config.map_name + '/sconv2/biases:0': vars[3],
            config.map_name + '/info_fc/weights:0': vars[4],
            config.map_name + '/info_fc/biases:0': vars[5],
            config.map_name + '/genFc/weights:0': vars[6],
            config.map_name + '/genFc/biases:0': vars[7],
            config.map_name + '/feat_fc/weights:0': vars[8],
            config.map_name + '/feat_fc/biases:0': vars[9],
            config.map_name + '/non_spatial_action/weights:0': vars[10],
            config.map_name + '/non_spatial_action/biases:0': vars[11],
            config.map_name + '/value/weights:0': vars[12],
            config.map_name + '/value/biases:0': vars[13]}

    return dict