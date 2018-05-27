import tensorflow as tf
import tensorflow.contrib.layers as layers
from base.base_model import BaseModel
import os


class StrategicNetwork(BaseModel):
    def __init__(self, config, scope_id):
        super(StrategicNetwork, self).__init__(config)
        self.number_of_actions = 5  # TODO solve number of actions
        self.scope_id = scope_id
        # self.init_saver()
        self.build_model()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        dict = {self.config.map_name + '/sconv1/weights:0': global_vars[0],
                self.config.map_name + '/sconv1/biases:0': global_vars[1],
                self.config.map_name + '/sconv2/weights:0': global_vars[2],
                self.config.map_name + '/sconv2/biases:0': global_vars[3],
                self.config.map_name + '/info_fc/weights:0': global_vars[4],
                self.config.map_name + '/info_fc/biases:0': global_vars[5],
                self.config.map_name + '/genFc/weights:0': global_vars[6],
                self.config.map_name + '/genFc/biases:0': global_vars[7],
                self.config.map_name + '/feat_fc/weights:0': global_vars[8],
                self.config.map_name + '/feat_fc/biases:0': global_vars[9],
                self.config.map_name + '/non_spatial_action/weights:0': global_vars[10],
                self.config.map_name + '/non_spatial_action/biases:0': global_vars[11],
                self.config.map_name + '/value/weights:0': global_vars[12],
                self.config.map_name + '/value/biases:0': global_vars[13]}

        self.saver = tf.train.Saver(dict, max_to_keep=self.config.max_to_keep)

    def save(self, sess):
        print("Saving model...")

        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.map_name, self.config.map_name +
                                           self.config.test_id + '.cptk'), self.global_step_tensor.eval())
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
            self.selected_action = tf.placeholder(tf.float32, [None, self.number_of_actions], name='action')

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

            # Non spatial action and value
            feat_fc = tf.concat([layers.flatten(sconv2), infoFc, genFc], axis=1)
            feat_fc = layers.fully_connected(feat_fc,
                                            num_outputs=256,
                                            activation_fn=tf.nn.relu,
                                            scope='feat_fc')

            # Output layers for policy and value estimations
            self.actionPolicy = layers.fully_connected(feat_fc,
                                                       num_outputs=self.number_of_actions,
                                                       activation_fn=tf.nn.softmax,
                                                       scope='non_spatial_action')

            self.value = tf.reshape(layers.fully_connected(feat_fc,
                                                           num_outputs=1,
                                                           activation_fn=None,
                                                           scope='value'), [-1])

    def init_worker_calc_variables(self):
        with tf.variable_scope(self.scope_id):
            # stores which actions were valid at a given time
            self.validActions = tf.placeholder(tf.float32, [None, self.number_of_actions], name="pVActions")
            # stores which action was selected at a given time
            self.selectedAction = tf.placeholder(tf.float32, [None, self.number_of_actions], name="pSActions")
            # stores the value we are aiming for
            self.valueTarget = tf.placeholder(tf.float32, [None], name="pVT")

            # policy(action|state)
            tempActionPolicy = tf.reduce_sum(self.actionPolicy * self.selectedAction, axis=1)
            validActionPolicy = tf.clip_by_value(tf.reduce_sum(self.actionPolicy * self.validActions), 1e-10, 1)
            validActionPolicy = tempActionPolicy / validActionPolicy
            validActionPolicy = tf.log(tf.clip_by_value(validActionPolicy, 1e-10, 1.))

            validPolicy = validActionPolicy

            # Gt - v(st) = advantage?
            advantage = tf.stop_gradient(self.valueTarget - self.value)
            self.policyLoss = - tf.reduce_mean(validPolicy * advantage)

            # self.valueLoss = - tf.reduce_mean(self.value * advantage)
            self.valueLoss = tf.losses.mean_squared_error(self.valueTarget, self.value)

            self.learningRate = tf.placeholder(tf.float32, None, name='learning_rate')
            # entropy regularization
            self.entropyLoss = - (tf.reduce_sum(self.actionPolicy * tf.log(self.actionPolicy)))
            self.loss = self.policyLoss \
                        + self.config.value_factor * self.valueLoss \
                        + self.config.entropy * self.entropyLoss

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
