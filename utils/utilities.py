import numpy as np
import tensorflow as tf
import sys, os


from pysc2.lib import features
from pysc2.lib import actions as scActions
from utils.email import notify_email



# Used to set the local worker network to that of the global network (copy)
def updateNetwork(fromNetwork,toNetwork):
    fromVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, fromNetwork)
    toVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, toNetwork)

    newNet = []
    for from_var,to_var in zip(fromVars,toVars):
        newNet.append(to_var.assign(from_var))
    return newNet


# Used to add any wanted feature layers
def addFeatureLayers(obs):
    featureLayers = np.array((obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index] /
                              features.SCREEN_FEATURES[features.SCREEN_FEATURES.player_relative.index].scale),
                             dtype=np.float32)

    featureLayers = np.expand_dims(featureLayers[np.newaxis], axis=0)

    featureLayers = np.insert(featureLayers, 0,
                              obs.observation["screen"][features.SCREEN_FEATURES.selected.index],
                              axis=1)


    featureLayers = np.insert(featureLayers, 0,
                              (obs.observation["screen"][features.SCREEN_FEATURES.player_id.index] /
                               features.SCREEN_FEATURES[features.SCREEN_FEATURES.player_id.index].scale),
                              axis=1)
    featureLayers = np.insert(featureLayers, 0,
                              (obs.observation["screen"][features.SCREEN_FEATURES.unit_type.index] /
                               features.SCREEN_FEATURES[features.SCREEN_FEATURES.unit_type.index].scale),
                              axis=1)
    featureLayers = np.insert(featureLayers, 0,
                              obs.observation["screen"][features.SCREEN_FEATURES.unit_hit_points.index] /
                              features.SCREEN_FEATURES[features.SCREEN_FEATURES.unit_hit_points.index].scale,
                              axis=1)
    featureLayers = np.insert(featureLayers, 0,
                              (obs.observation["screen"][features.SCREEN_FEATURES.unit_density.index] /
                               features.SCREEN_FEATURES[features.SCREEN_FEATURES.unit_density.index].scale),
                              axis=1)
    featureLayers = np.insert(featureLayers, 0,
                              (obs.observation["screen"][features.SCREEN_FEATURES.unit_density_aa.index] /
                               features.SCREEN_FEATURES[features.SCREEN_FEATURES.unit_density_aa.index].scale), axis=1)
    # featureLayers = np.insert(featureLayers, 0,
    #   obs.observation["screen"][features.SCREEN_FEATURES.unit_type.index], axis=1)

    return featureLayers


def addGeneralFeatures(obs):
    genFeat = obs.observation["player"]
    buildQueue = obs.observation["build_queue"]
    singleSelect = obs.observation["single_select"]

    multiSelect = np.array(obs.observation["multi_select"], dtype=np.float32)
    selection = np.zeros([1, 20, 7])
    featureLayers = np.zeros([1, 11], dtype=np.float32)
    bqueue = np.zeros([1, 5, 7], dtype=np.float32)

    if singleSelect != []:
        selection[0][0] = singleSelect
    elif multiSelect != []:
        for i in range(len(multiSelect)):
            if i == 20:
                break
            selection[0][i] = multiSelect[i]

    for i in range(len(buildQueue)):
        bqueue[0][i] = buildQueue[i]
    for i in range(len(genFeat)) :
        featureLayers[0, i] = genFeat[i]

    return featureLayers, bqueue, selection


# Used to add any wanted actions
def getAvailableActions(obs, map_name):
    availActions = obs.observation['available_actions']
    chosenActions = []

    if map_name == "HHAssignSCV":
        chosenActions = getAvailableActionsASCV(obs)
    elif map_name == "CollectMineralShards":
        chosenActions = getAvailableActionsCMS(obs)
    elif map_name == "HHExpandArmy2":
        chosenActions = getAvailableActionsEA(obs)
    elif map_name == "HHBuildBarracks":
        chosenActions = getAvailableActionsBB(obs)
    elif map_name == "HHBuildSCV":
        chosenActions = getAvailableActionsBSCV(obs)
    elif map_name == "HHBuildSupply":
        chosenActions = getAvailableActionsBS(obs)
    else:
        raise Exception("Action set not defined for the map '" + map_name + "'. Also check spelling.")

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsCMS(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsASCV(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)
    #chosenActions.append(scActions.FUNCTIONS.select_rect.id)
    #chosenActions.append(scActions.FUNCTIONS.select_army.id)
    chosenActions.append(scActions.FUNCTIONS.select_idle_worker.id)
    #chosenActions.append(scActions.FUNCTIONS.build_queue.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Refinery_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Barracks_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_CommandCenter_screen.id)
    chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_screen.id)
    chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_SCV_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_SupplyDepot_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_Workers_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_Marine_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_SCV_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.select_control_group.id)

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsBSCV(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)
    #chosenActions.append(scActions.FUNCTIONS.select_rect.id)
    #chosenActions.append(scActions.FUNCTIONS.select_army.id)
    #chosenActions.append(scActions.FUNCTIONS.select_idle_worker.id)
    #chosenActions.append(scActions.FUNCTIONS.build_queue.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Refinery_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Barracks_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_SCV_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_SupplyDepot_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_Workers_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_Marine_quick.id)
    chosenActions.append(scActions.FUNCTIONS.Train_SCV_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.select_control_group.id)

    return list(set(availActions) & set(chosenActions))


def getAvailableActionsBB(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)
    #chosenActions.append(scActions.FUNCTIONS.select_rect.id)
    #chosenActions.append(scActions.FUNCTIONS.select_army.id)
    chosenActions.append(scActions.FUNCTIONS.select_idle_worker.id)
    #chosenActions.append(scActions.FUNCTIONS.build_queue.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Refinery_screen.id)
    chosenActions.append(scActions.FUNCTIONS.Build_Barracks_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_SCV_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_SupplyDepot_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_Workers_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_Marine_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_SCV_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.select_control_group.id)

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsBS(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)
    #chosenActions.append(scActions.FUNCTIONS.select_rect.id)
    #chosenActions.append(scActions.FUNCTIONS.select_army.id)
    chosenActions.append(scActions.FUNCTIONS.select_idle_worker.id)
    #chosenActions.append(scActions.FUNCTIONS.build_queue.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Refinery_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Barracks_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_SCV_screen.id)
    chosenActions.append(scActions.FUNCTIONS.Build_SupplyDepot_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_Workers_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_Marine_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_SCV_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.select_control_group.id)

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsEA(obs):
    availActions = obs.observation['available_actions']
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.no_op.id)
    chosenActions.append(scActions.FUNCTIONS.select_point.id)
    #chosenActions.append(scActions.FUNCTIONS.select_rect.id)
    #chosenActions.append(scActions.FUNCTIONS.select_army.id)
    #chosenActions.append(scActions.FUNCTIONS.select_idle_worker.id)
    #chosenActions.append(scActions.FUNCTIONS.build_queue.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Refinery_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_Barracks_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_CommandCenter_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Harvest_Gather_SCV_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Build_SupplyDepot_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_Workers_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Rally_CommandCenter_screen.id)
    chosenActions.append(scActions.FUNCTIONS.Train_Marine_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Train_SCV_quick.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.Attack_Attack_screen.id)
    #chosenActions.append(scActions.FUNCTIONS.select_control_group.id)

    return list(set(availActions) & set(chosenActions))

def getAvailableActionsStrat(obs):
    chosenActions = []

    #chosenActions.append(scActions.FUNCTIONS.Move_screen.id)
    chosenActions.append(0)
    chosenActions.append(1)
    chosenActions.append(2)
    chosenActions.append(3)
    chosenActions.append(4)

    return chosenActions

def terminate(config):
    print("Terminating program")

    if config.notify_by_email:
        notify_email()

    if config.shutdown:
       os.system("shutdown -h -t 59")
       sys.exit()
    else:
        sys.exit()
