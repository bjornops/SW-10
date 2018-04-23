import tensorflow as tf
import tensorflow.contrib.layers as layers
from base.base_model import BaseModel
import os


class TacticalNetwork(BaseModel):
    def __init__(self, config, scope_id):
        super(TacticalNetwork, self).__init__(config)
        self.number_of_actions = 524  # TODO solve number of actions
        self.scope_id = scope_id
        self.init_saver()
        self.build_model()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def save(self, sess):
        print("Saving model...")
        globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        dict = {self.config.map_name + '/sconv1/weights:0': globalVars[0],
                self.config.map_name + '/sconv1/biases:0': globalVars[1],
                self.config.map_name + '/sconv2/weights:0': globalVars[2],
                self.config.map_name + '/sconv2/biases:0': globalVars[3],
                self.config.map_name + '/info_fc/weights:0': globalVars[4],
                self.config.map_name + '/info_fc/biases:0': globalVars[5],
                self.config.map_name + '/genFc/weights:0': globalVars[6],
                self.config.map_name + '/genFc/biases:0': globalVars[7],
                self.config.map_name + '/spPol/weights:0': globalVars[8],
                self.config.map_name + '/spPol/biases:0': globalVars[9],
                self.config.map_name + '/feat_fc/weights:0': globalVars[10],
                self.config.map_name + '/feat_fc/biases:0': globalVars[11],
                self.config.map_name + '/non_spatial_action/weights:0': globalVars[12],
                self.config.map_name + '/non_spatial_action/biases:0': globalVars[13],
                self.config.map_name + '/value/weights:0': globalVars[14],
                self.config.map_name + '/value/biases:0': globalVars[15]}
        saver = tf.train.Saver(dict)
        saver.save(sess, self.config.checkpoint_dir + "/" + self.config.map_name + "/" + self.config.map_name + self.config.test_id + '.cptk', self.global_step_tensor.eval())
        print("Model Saved")

    def build_model(self):
        with tf.variable_scope(self.scope_id):
            # Extract features
            self.screen = tf.placeholder(tf.float32, [None, self.config.feature_count, self.config.screen_size,
                                                      self.config.screen_size], name='screen')
            self.actionInfo = tf.placeholder(tf.float32, [None, self.number_of_actions], name='actionInfo')
            self.generalFeatures = tf.placeholder(tf.float32, [None, 11], name='generalFeatures')
            self.buildQueue = tf.placeholder(tf.float32, [None, 5, 7], name='bQueue')
            self.selection = tf.placeholder(tf.float32, [None, 20, 7], name='selection')

            sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='sconv1')
            sconv2 = layers.conv2d(sconv1,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='sconv2')

            infoFc = layers.fully_connected(tf.concat([layers.flatten(self.actionInfo)],
                                                      axis=1),
                                            num_outputs=128,
                                            activation_fn=tf.tanh,
                                            scope='info_fc')

            genFc = layers.fully_connected(tf.concat([layers.flatten(self.generalFeatures),
                                                      layers.flatten(self.buildQueue),
                                                      layers.flatten(self.selection)],
                                                     axis=1),
                                           num_outputs=128,
                                           activation_fn=tf.tanh,
                                           scope='genFc')

            # Spatial action
            spatialPolicy = layers.conv2d(sconv2,
                                          num_outputs=1,
                                          kernel_size=1,
                                          stride=1,
                                          activation_fn=None,
                                          scope='spPol')

            self.spatialPolicy = tf.nn.softmax(layers.flatten(spatialPolicy))

            # Non spatial action and value
            featFc = tf.concat([layers.flatten(sconv2), infoFc, genFc], axis=1)
            featFc = layers.fully_connected(featFc,
                                            num_outputs=256,
                                            activation_fn=tf.nn.relu,
                                            scope='feat_fc')

            # Output layers for policy and value estimations
            self.actionPolicy = layers.fully_connected(featFc,
                                                       num_outputs=self.number_of_actions,
                                                       activation_fn=tf.nn.softmax,
                                                       scope='non_spatial_action')

            self.value = tf.reshape(layers.fully_connected(featFc,
                                                           num_outputs=1,
                                                           activation_fn=None,
                                                           scope='value'), [-1])

    def init_worker_calc_variables(self):
        with tf.variable_scope(self.scope_id):
            # stores which actions were valid at a given time
            self.validActions = tf.placeholder(tf.float32, [None, self.number_of_actions], name="pVActions")
            # stores which action was selected at a given time
            self.selectedAction = tf.placeholder(tf.float32, [None, self.number_of_actions], name="pSActions")
            # stores the picked spatial
            self.selectedSpatialAction = tf.placeholder(tf.float32, [None, self.config.screen_size**2], name="pSPActions")
            # used for storing whether the current action made use of a spatial action
            self.validSpatialAction = tf.placeholder(tf.float32, [None], name="pVSActions")
            # stores the value we are aiming for
            self.valueTarget = tf.placeholder(tf.float32, [None], name="pVT")


            ##calc
            # policy(spatial|state)
            tempSpatialPolicy = tf.reduce_sum(self.spatialPolicy * self.selectedSpatialAction, axis=1)
            # log(policy(spatial|state))
            logOfSpatialPolicy = tf.log(tf.clip_by_value(tempSpatialPolicy, 1e-10, 1.))
            # we use clip by value to ensure no v <= 0 and v > 1 values
            validSpatialPolicy = logOfSpatialPolicy * self.validSpatialAction

            # policy(action|state)
            tempActionPolicy = tf.reduce_sum(self.actionPolicy * self.selectedAction, axis=1)
            validActionPolicy = tf.clip_by_value(tf.reduce_sum(self.actionPolicy * self.validActions), 1e-10, 1)
            validActionPolicy = tempActionPolicy / validActionPolicy
            validActionPolicy = tf.log(tf.clip_by_value(validActionPolicy, 1e-10, 1.))

            validPolicy = validActionPolicy + validSpatialPolicy

            # Gt - v(st) = advantage?
            advantage = tf.stop_gradient(self.valueTarget - self.value)
            self.policyLoss = - tf.reduce_mean(validPolicy * advantage)

            self.valueLoss = - tf.reduce_mean(self.value * advantage)

            self.learningRate = tf.placeholder(tf.float32, None, name='learning_rate')
            # entropy regularization
            self.entropyLoss = - (tf.reduce_sum(self.actionPolicy * tf.log(self.actionPolicy)) +
                                  tf.reduce_sum(self.spatialPolicy * tf.log(self.spatialPolicy)))
            self.loss = self.policyLoss + self.config.value_factor * self.valueLoss + self.config.entropy * self.entropyLoss

            optimizer = tf.train.RMSPropOptimizer(self.learningRate, decay=0.99, epsilon=1e-10)
            # Get gradients from local network using local losses
            localVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_id)
            self.gradients = optimizer.compute_gradients(self.loss, localVars)
            self.varNorms = tf.global_norm(localVars)

            self.grads = []
            for grad, _ in self.gradients:
                grad = tf.clip_by_norm(grad, 10.0)
                self.grads.append(grad)
            # Apply local gradients to global network
            globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.applyGrads = optimizer.apply_gradients(zip(self.grads, globalVars))
