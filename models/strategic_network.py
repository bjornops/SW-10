import tensorflow as tf
import tensorflow.contrib.layers as layers
from base.base_model import BaseModel


class StrategicNetwork(BaseModel):
    def __init__(self, config, scope_id):
        super(StrategicNetwork, self).__init__(config)
        self.build_model(number_of_actions=5, scope_id=scope_id)
        self.init_saver()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self, number_of_actions, scope_id):
        with tf.variable_scope(scope_id):
            # Extract features
            self.screen = tf.placeholder(tf.float32, [None, self.config.feature_count, self.config.screen_size,
                                                      self.config.screen_size], name='screen')
            self.actionInfo = tf.placeholder(tf.float32, [None, number_of_actions], name='actionInfo')
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


            infoFc = layers.fully_connected(layers.flatten(self.actionInfo),
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


            #Non spatial action and value
            featFc = tf.concat([layers.flatten(sconv2), infoFc, genFc], axis=1)
            featFc = layers.fully_connected(featFc,
                                            num_outputs=256,
                                            activation_fn=tf.nn.relu,
                                            scope='feat_fc')

            #Output layers for policy and value estimations
            self.actionPolicy = layers.fully_connected(featFc,
                                                       num_outputs=number_of_actions,
                                                       activation_fn=tf.nn.softmax,
                                                       scope='non_spatial_action')

            self.value = tf.reshape(layers.fully_connected(featFc,
                                                           num_outputs=1,
                                                           activation_fn=None,
                                                           scope='value'), [-1])

            #Only the worker network need ops for loss functions and gradient updating.
            if scope_id != 'global':

                # stores which actions were valid at a given time
                self.validActions = tf.placeholder(tf.float32, [None, number_of_actions], name="pVActions")
                # stores which action was selected at a given time
                self.selectedAction = tf.placeholder(tf.float32, [None, number_of_actions], name="pSActions")
                # stores the picked spatial actions
                self.selectedSpatialAction = tf.placeholder(tf.float32, [None, self.config.screen_size**2], name="pSPActions")
                # used for storing whether the current action made use of a spatial action
                self.validSpatialAction = tf.placeholder(tf.float32, [None], name="pVSActions")
                # stores the value we are aiming for
                self.valueTarget = tf.placeholder(tf.float32, [None], name="pVT")

                ## calc

                # policy(action|state)
                tempActionPolicy = tf.reduce_sum(self.actionPolicy * self.selectedAction, axis=1)
                validActionPolicy = tf.clip_by_value(tf.reduce_sum(self.actionPolicy * self.validActions), 1e-10, 1)
                validActionPolicy = tempActionPolicy / validActionPolicy
                validActionPolicy = tf.log(tf.clip_by_value(validActionPolicy, 1e-10, 1.))

                validPolicy = validActionPolicy

                # Gt - v(st) = advantage
                advantage = tf.stop_gradient(self.valueTarget - self.value)
                self.policyLoss = - tf.reduce_mean(validPolicy * advantage)

                self.valueLoss = - tf.reduce_mean(self.value * advantage)

                self.learningRate = tf.placeholder(tf.float32, None, name='learning_rate')
                # entropy regularization
                self.entropyLoss = - (tf.reduce_sum(self.actionPolicy * tf.log(self.actionPolicy)))
                self.loss = self.policyLoss + self.config.value_factor * self.valueLoss + self.config.entropy * self.entropyLoss

                optimizer = tf.train.RMSPropOptimizer(self.learningRate, decay=0.99, epsilon=1e-10)
                # Get gradients from local network using local losses
                localVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_id)
                self.gradients = optimizer.compute_gradients(self.loss, localVars)
                self.varNorms = tf.global_norm(localVars)
                grads = []
                for grad, _ in self.gradients:
                    grad = tf.clip_by_norm(grad, 10.0)
                    grads.append(grad)
                # Apply local gradients to global network
                globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.applyGrads = optimizer.apply_gradients(zip(grads, globalVars))