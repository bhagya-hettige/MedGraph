import tensorflow as tf
import numpy as np
from utils import sparse_feeder

seed = 46


class MedGraph:
    def __init__(self, dataLoader):
        # Hyperparameters
        alpha = 1
        beta = 1
        gamma = 1

        # Seed for reproducibility
        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Data types for tensors
        self.INT_TYPE = tf.int32
        self.FLOAT_TYPE = tf.float32

        # Parameter dimensions
        self.N_visits, self.D_visits = dataLoader.X1.shape
        self.N_codes, self.D_codes = dataLoader.X2.shape
        self.sizes = [self.D_visits, self.D_codes]
        self.L = dataLoader.embedding_dim
        self.n_hidden = [512]
        self.rnn_hidden = [128]

        self.X_visits = tf.sparse_placeholder(name='X_visits', dtype=self.FLOAT_TYPE)
        self.X = [self.X_visits, tf.SparseTensor(*sparse_feeder(dataLoader.X_codes))]

        self.alpha = tf.placeholder(name='alpha', dtype=self.FLOAT_TYPE)
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=self.FLOAT_TYPE)

        # v-c graph
        self.vc_u_i = tf.placeholder(name='vc_u_i', dtype=self.INT_TYPE,
                                     shape=[dataLoader.batch_size_vc * (dataLoader.K + 1)])
        self.vc_u_j = tf.placeholder(name='vc_u_j', dtype=self.INT_TYPE,
                                     shape=[dataLoader.batch_size_vc * (dataLoader.K + 1)])
        self.vc_label = tf.placeholder(name='vc_label', dtype=self.FLOAT_TYPE,
                                       shape=[dataLoader.batch_size_vc * (dataLoader.K + 1)])

        # v->v sequences
        self.vv_in_time = tf.placeholder(name='vv_in_time', dtype=self.FLOAT_TYPE, shape=[None, None])
        self.vv_out_time = tf.placeholder(name='vv_out_time', dtype=self.FLOAT_TYPE, shape=[None, None])
        self.vv_out_mask = tf.placeholder(name='vv_out_mask', dtype=self.INT_TYPE, shape=[None, None, 1])
        self.vv_inputs = tf.placeholder(name='vv_inputs', dtype=self.INT_TYPE, shape=(None, None))
        self.vv_outputs = tf.placeholder(name='vv_outputs', dtype=self.FLOAT_TYPE, shape=(None, None))

        # Create model architecture
        self.__create_model()

        L_struc = self.structural_loss_gauss(self.vc_u_i, self.vc_u_j, self.vc_label, dataLoader.distance) \
            if dataLoader.is_gauss else self.structural_loss_inner(self.vc_u_i, self.vc_u_j, self.vc_label)

        L_reg = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + self.sigma[0] - tf.square(self.embedding[0]) - tf.exp(self.sigma[0]), axis=1)) + \
                tf.reduce_mean(-0.5 * tf.reduce_sum(
                    1 + self.sigma[1] - tf.square(self.embedding[1]) - tf.exp(self.sigma[1]), axis=1)) \
            if dataLoader.is_gauss else 0

        L_temp, L_aux = self.__temporal_loss(dataLoader.is_gauss, dataLoader.is_time_dis)

        self.loss = alpha * L_struc + beta * 1e-4 * L_temp + gamma * L_aux + 1e-5 * L_reg

        self.optimizer = tf.train.AdamOptimizer(learning_rate=dataLoader.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def __create_model(self):
        w_init = tf.contrib.layers.xavier_initializer

        # Create attribute encoders to transform visits and codes into the same latent space
        self.encoded = [self.create_attribute_encoder(idx) for idx in range(len(self.sizes))]

        # Create Gaussian embedding for visit and code nodes using the encoded representations
        W_mu = tf.get_variable(name='W_mu', shape=[self.n_hidden[-1], self.L], dtype=self.FLOAT_TYPE,
                               initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=self.FLOAT_TYPE, initializer=w_init())
        W_sigma = tf.get_variable(name='W_sigma', shape=[self.n_hidden[-1], self.L], dtype=self.FLOAT_TYPE,
                                  initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=self.FLOAT_TYPE,
                                  initializer=w_init())
        self.embedding = [tf.matmul(self.encoded[idx], W_mu) + b_mu for idx in range(len(self.sizes))]
        self.sigma = [tf.nn.elu(tf.matmul(self.encoded[idx], W_sigma) + b_sigma) + 1 + 1e-14
                      for idx in range(len(self.sizes))]

    def create_attribute_encoder(self, index=0):
        sizes = [self.sizes[index]] + self.n_hidden
        w_init = tf.contrib.layers.xavier_initializer
        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_enc{}{}'.format(index, i), shape=[sizes[i - 1], sizes[i]],
                                dtype=self.FLOAT_TYPE,
                                initializer=w_init())
            b = tf.get_variable(name='b_enc{}{}'.format(index, i), shape=[sizes[i]], dtype=self.FLOAT_TYPE,
                                initializer=w_init())
            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X[index], W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)
        return encoded

    def structural_loss_gauss(self, u_i, u_j, label, distance):
        energy = -self.energy_kl(u_i, u_j) if distance == 'kl' else -self.energy_w2(u_i, u_j)
        loss = -tf.reduce_mean(tf.log_sigmoid(label * energy))
        return loss

    def structural_loss_inner(self, u_i, u_j, label):
        embedding = tf.concat(self.embedding, axis=0)
        u_i_embedding = tf.gather(embedding, u_i)
        u_j_embedding = tf.gather(embedding, u_j)
        inner_product = tf.reduce_sum(u_i_embedding * u_j_embedding, axis=1)
        loss = -tf.reduce_mean(tf.log_sigmoid(label * inner_product))
        return loss

    def __temporal_loss(self, is_gauss=True, is_time_dis=True):
        n_classes = tf.shape(self.vv_outputs)[1]
        batch_size = tf.shape(self.vv_in_time)[0]

        # Mask for variable length visit sequences
        mask = tf.cast(tf.sign(tf.reduce_max(tf.abs(self.vv_out_mask), 2)), self.FLOAT_TYPE)

        # Get the embedding of the visits as markers
        if is_gauss:
            eps = tf.random_normal(shape=tf.shape(self.sigma[0]), mean=0, stddev=1, dtype=self.FLOAT_TYPE)
            z = self.embedding[0] + tf.sqrt(tf.exp(self.sigma[0])) * eps
        else:
            z = tf.gather(self.embedding[0], self.vv_inputs)
        guessed_z = tf.gather(z, self.vv_inputs)
        # Apply masking for padded visits
        vv_in_mask = tf.tile(self.vv_out_mask, [1, 1, self.L])
        comparison = tf.equal(vv_in_mask, tf.constant(0))
        guessed_z = tf.where(comparison, tf.zeros_like(guessed_z), guessed_z)

        # Calculate time gaps between consecutive visits
        last_time = tf.concat([tf.zeros([batch_size, 1], dtype=self.FLOAT_TYPE), self.vv_in_time[:, :-1]], axis=1)
        delta_t_prev = tf.expand_dims(self.vv_in_time - last_time, -1)
        delta_t_next = tf.expand_dims(self.vv_out_time - self.vv_in_time, -1)

        # Append time gap information at the end of the marker information
        guessed_z = tf.concat([guessed_z, delta_t_prev], axis=2)

        # Create the RNN cell
        cells = []
        for n in self.rnn_hidden:
            cell = tf.contrib.rnn.LSTMCell(n)
            # cell = tf.contrib.rnn.DropoutWrapper(
            #     cell, output_keep_prob=0.95)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Get RNN outputs for the visit sequences
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, guessed_z, dtype=self.FLOAT_TYPE,
                                           sequence_length=self._get_vv_sequence_length(guessed_z))
        # Apply masking for visit sequences
        vv_out_mask = tf.tile(self.vv_out_mask, [1, 1, self.rnn_hidden[-1]])
        output = tf.reshape(tf.boolean_mask(rnn_outputs, vv_out_mask), [-1, self.rnn_hidden[-1]])
        weight, bias = self._weight_and_bias(self.rnn_hidden[-1], n_classes)

        # Flatten to apply same weights to all time steps
        output = tf.reshape(output, [-1, self.rnn_hidden[-1]])
        self.y = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.y_class = tf.argmax(self.y, 1)

        # last_output = self._last_relevant(rnn_outputs, self.length(guessed_z))
        # self.y_last = tf.nn.softmax(tf.matmul(last_output, weight) + bias)
        # self.y_last_class = tf.argmax(self.y_last, 1)

        # Auxiliary task loss
        sup_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.y, onehot_labels=self.vv_outputs))

        # Temporal loss based on marked point processes
        rnn_outputs_shape = tf.shape(rnn_outputs)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_outputs_shape[-1]])

        self.W_t = tf.get_variable('Wt', (self.rnn_hidden[-1], 1),
                                   initializer=tf.constant_initializer(np.ones((self.rnn_hidden[-1], 1)) * 0.001))
        self.w_t = tf.get_variable('wt', (1), initializer=tf.constant_initializer(1.0))
        self.b_t = tf.get_variable('bt', (1), initializer=tf.constant_initializer(np.log(1.0)))

        wt = tf.nn.softplus(self.w_t)
        delta_t_next = tf.reshape(delta_t_next, [-1, 1])
        part1 = tf.matmul(rnn_outputs, self.W_t) + self.b_t
        part2 = (-wt * delta_t_next)
        time_loglike = part1 + part2 + (tf.exp(tf.minimum(50.0, part1 + part2)) - tf.exp(tf.minimum(50.0, part1))) / wt
        total_loss = time_loglike

        # Apply masking for variable length visit sequences
        total_loss = tf.reshape(total_loss, [batch_size, -1])
        total_loss *= mask

        # Average over actual sequence lengths
        total_loss = tf.reduce_sum(total_loss, axis=1)
        vv_loss = -tf.reduce_mean(total_loss)

        return vv_loss, sup_loss

    def get_vv_sequence_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, self.INT_TYPE)
        return length

    def energy_kl(self, u_i, u_j):
        embedding = tf.concat(self.embedding, axis=0)
        sigma = tf.concat(self.sigma, axis=0)

        mu_i = tf.gather(embedding, u_i)
        sigma_i = tf.gather(sigma, u_i)
        mu_j = tf.gather(embedding, u_j)
        sigma_j = tf.gather(sigma, u_j)

        sigma_ratio = sigma_j / sigma_i
        trace_fac = tf.reduce_sum(sigma_ratio, axis=1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), axis=1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, axis=1)

        ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        sigma_ratio = sigma_i / sigma_j
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

        ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        return 0.5 * (ij_kl + ji_kl)

    def energy_w2(self, u_i, u_j):
        embedding = tf.concat(self.embedding, axis=0)
        sigma = tf.concat(self.sigma, axis=0)

        mu_i = tf.gather(embedding, u_i)
        sigma_i = tf.gather(sigma, u_i)
        mu_j = tf.gather(embedding, u_j)
        sigma_j = tf.gather(sigma, u_j)

        delta = mu_i - mu_j
        d1 = tf.reduce_sum(delta * delta, axis=1)
        x0 = sigma_i - sigma_j
        d2 = tf.reduce_sum(x0 * x0, axis=1)
        wd = d1 + d2

        return wd
