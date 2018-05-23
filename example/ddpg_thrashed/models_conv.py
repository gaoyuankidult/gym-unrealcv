import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            init_conv = tc.layers.xavier_initializer_conv2d(uniform=True)
            init_dense = tc.layers.xavier_initializer(uniform=True)
            init_output = tf.random_uniform_initializer(minval=-3e-5, maxval=3e-5)
            training_mode = tf.get_default_graph().get_tensor_by_name("training_mode:0")

            x = obs
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn1'
            #                       )
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.conv2d(x, 32, (5,5), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn2'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)


            ''' Took one out for mem-reasons '''
            x = tf.layers.conv2d(x, 32, (3,3), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn3'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)


            x = tf.layers.conv2d(x, 64, (3,3), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn4'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tc.layers.flatten(x)
            x = tf.layers.dense(x, 512, kernel_initializer=init_dense)
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn5'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.dense(x, 512, kernel_initializer=init_dense)
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn6'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=init_output)
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            init_conv = tc.layers.xavier_initializer_conv2d(uniform=True)
            init_dense = tc.layers.xavier_initializer(uniform=True)
            init_output = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            training_mode = tf.get_default_graph().get_tensor_by_name("training_mode:0")

            x = obs
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn7'
            #                       )
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.conv2d(x, 32, (5,5), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn8'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.conv2d(x, 32, (3,3), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn9'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.conv2d(x, 64, (3,3), strides=(1,1), padding='same', kernel_initializer=init_conv)
            x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), padding='same', data_format='channels_last')
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn10'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tc.layers.flatten(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 512, kernel_initializer=init_dense)
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn11'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.dense(x, 512, kernel_initializer=init_dense)
            # x = tc.layers.batch_norm(x,
            #                       center=True, scale=True,
            #                       reuse=tf.AUTO_REUSE,
            #                       is_training=training_mode,
            #                       scope='bn12'
            #                       )
            x = tf.nn.elu(x)
            x = tf.layers.dropout(x, rate=0.2)

            x = tf.layers.dense(x, 1, kernel_initializer=init_output)
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
