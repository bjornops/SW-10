import tensorflow as tf
import tensorflow.contrib.layers as layers

class TacticalNetwork():
    def __init__(self, screenSize, numberOfActions, scope, numberOfFeatures, beta, eta):
        with tf.variable_scope(scope):
            # Extract features
            self.screen = tf.placeholder(tf.float32, [None, numberOfFeatures, screenSize, screenSize], name='screen')
            self.actionInfo = tf.placeholder(tf.float32, [None, numberOfActions], name='actionInfo')
            self.generalFeatures = tf.placeholder(tf.float32, [None, 11], name='generalFeatures')
            self.buildQueue = tf.placeholder(tf.float32, [None, 5, 7], name='bQueue')
            self.selection = tf.placeholder(tf.float32, [None, 20, 7], name='selection')
            self.previousActions = tf.placeholder(tf.float32, [None, 1], name='previousActions')

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


            infoFc = layers.fully_connected(tf.concat([layers.flatten(self.actionInfo),
                                                       layers.flatten(self.previousActions)],
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
                                                       num_outputs=numberOfActions,
                                                       activation_fn=tf.nn.softmax,
                                                       scope='non_spatial_action')

            self.value = tf.reshape(layers.fully_connected(featFc,
                                                           num_outputs=1,
                                                           activation_fn=None,
                                                           scope='value'), [-1])

            # Only the worker network need ops for loss functions and gradient updating. We also exclude tactical networks for hier
            if scope != 'global' and scope != 'HHExpandArmy2' and scope != 'HHBuildBarracks' and scope != 'HHAssignSCV' and scope != 'HHBuildSupply' and scope != 'HHBuildSCV':

                # stores which actions were valid at a given time
                self.validActions = tf.placeholder(tf.float32, [None, numberOfActions], name="pVActions")
                # stores which action was selected at a given time
                self.selectedAction = tf.placeholder(tf.float32, [None, numberOfActions], name="pSActions")
                # stores the picked spatial
                self.selectedSpatialAction = tf.placeholder(tf.float32, [None, screenSize**2], name="pSPActions")
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
                self.loss = self.policyLoss + beta * self.valueLoss + eta * self.entropyLoss

                optimizer = tf.train.RMSPropOptimizer(self.learningRate, decay=0.99, epsilon=1e-10)

                optimizer = tf.train.RMSPropOptimizer(self.learningRate, decay=0.99, epsilon=1e-10)
                # Get gradients from local network using local losses
                localVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = optimizer.compute_gradients(self.loss, localVars)
                self.varNorms = tf.global_norm(localVars)
                grads = []
                for grad, _ in self.gradients:
                    grad = tf.clip_by_norm(grad, 10.0)
                    grads.append(grad)
                # Apply local gradients to global network
                globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.applyGrads = optimizer.apply_gradients(zip(grads, globalVars))