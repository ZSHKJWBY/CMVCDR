import tensorflow as tf
import numpy as np


class OurNet(object):
    def __init__(self, user_count, item_count_1, item_count_2, lr, embedding_size=32, hidden_size=64, intra_dim=16,
                 num_layers=1, ssl_reg=0.001, ssl_temp=0.1, max_length_each_step=20, agg_mode='concat'):
        self.user_count = user_count
        self.item_count_1 = item_count_1
        self.item_count_2 = item_count_2
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.intra_dim = intra_dim
        self.ssl_reg = ssl_reg
        self.ssl_temp = ssl_temp
        self.agg_mode = agg_mode

        self.max_length_each_step = max_length_each_step
        self.max_interacted_item = max_length_each_step

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.u_1 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_1 = tf.placeholder(dtype=tf.int32, shape=[None, None])  # [batch_size, max_length]
        self.len_1 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.target_1 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.y_1 = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.hist_cross_item_1 = tf.placeholder(dtype=tf.int32, shape=[None, None, None])
        # [batch_size, max_length, each_step_max_length]

        self.u_2 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_2 = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.len_2 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.target_2 = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.y_2 = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.hist_cross_item_2 = tf.placeholder(dtype=tf.int32, shape=[None, None, None])

        specific_user_embedding_1 = tf.get_variable('specific_user_embedding_1',
                                                    shape=[self.user_count, self.embedding_size],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        specific_user_embedding_2 = tf.get_variable('specific_user_embedding_2',
                                                    shape=[self.user_count, self.embedding_size],
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        invariant_user_embedding_1 = tf.get_variable('invariant_user_embedding_1',
                                                     shape=[self.user_count, self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        invariant_user_embedding_2 = tf.get_variable('invariant_user_embedding_2',
                                                     shape=[self.user_count, self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        item_embedding_1 = tf.get_variable('item_embedding_1', shape=[self.item_count_1, self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        item_embedding_2 = tf.get_variable('item_embedding_2', shape=[self.item_count_2, self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        # self.input_1 & self.input_2: [batch_size, max_length]
        # input_item_embed_1 & input_item_embed_2: [batch_size, max_length, embedding_size]
        # self.target_1 & self.target_1: [batch_size, ]
        # target_item_embed_1 & target_item_embed_2: [batch_size, embedding_size]
        input_item_embed_1 = tf.nn.embedding_lookup(item_embedding_1, self.input_1)
        input_item_embed_2 = tf.nn.embedding_lookup(item_embedding_2, self.input_2)
        target_item_embed_1 = tf.nn.embedding_lookup(item_embedding_1, self.target_1)
        target_item_embed_2 = tf.nn.embedding_lookup(item_embedding_2, self.target_2)

        # cross_item_embed_1 & cross_item_embed_2: [batch_size, max_length, each_step_max_length, embedding_size]
        cross_item_embed_1 = self.generate_mask_item_embed(item_embedding_2, self.hist_cross_item_1)
        cross_item_embed_2 = self.generate_mask_item_embed(item_embedding_1, self.hist_cross_item_2)
        # cross_item_embed_1 = tf.nn.embedding_lookup(item_embedding_2, self.hist_cross_item_1)
        # cross_item_embed_2 = tf.nn.embedding_lookup(item_embedding_1, self.hist_cross_item_2)

        # 自监督相关
        # 域1中的用户在域1中的域特定兴趣表示
        specific_user_1_for_1 = tf.nn.embedding_lookup(specific_user_embedding_1, self.u_1)
        # 域1中的用户在域2中的域特定兴趣表示
        specific_user_2_for_1 = tf.nn.embedding_lookup(specific_user_embedding_2, self.u_1)
        invariant_user_1_for_1 = tf.nn.embedding_lookup(invariant_user_embedding_1, self.u_1)
        invariant_user_2_for_1 = tf.nn.embedding_lookup(invariant_user_embedding_2, self.u_1)

        specific_user_1_for_2 = tf.nn.embedding_lookup(specific_user_embedding_1, self.u_2)
        specific_user_2_for_2 = tf.nn.embedding_lookup(specific_user_embedding_2, self.u_2)
        invariant_user_1_for_2 = tf.nn.embedding_lookup(invariant_user_embedding_1, self.u_2)
        invariant_user_2_for_2 = tf.nn.embedding_lookup(invariant_user_embedding_2, self.u_2)

        # # each_step_cross_embed_1 & each_step_cross_embed_2:
        # [batch_size, max_length, embedding_size]
        in_dim = self.embedding_size * 3
        out_dim = self.hidden_size
        weight_matrix_1_1 = tf.get_variable('weight_matrix_1_1', shape=[in_dim, out_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        weight_matrix_2_1 = tf.get_variable('weight_matrix_2_1', shape=[in_dim, out_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        in_dim = self.hidden_size
        out_dim = 1
        weight_matrix_1_2 = tf.get_variable('weight_matrix_1_2', shape=[in_dim, out_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        weight_matrix_2_2 = tf.get_variable('weight_matrix_2_2', shape=[in_dim, out_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        # [batch_size, max_length, embedding_size]->[batch_size*max_length*max_num_each_step, embedding_size]
        invariant_representation_as_query_1 = self.reshape_item_embed(invariant_user_1_for_1, self.max_length_each_step,
                                                                      self.max_interacted_item)
        invariant_representation_as_query_2 = self.reshape_item_embed(invariant_user_2_for_2, self.max_length_each_step,
                                                                      self.max_interacted_item)
        each_step_cross_embed_1 = self.agg_by_attention(cross_item_embed_1, invariant_representation_as_query_1,
                                                        weight_matrix_1_1, weight_matrix_1_2)
        each_step_cross_embed_2 = self.agg_by_attention(cross_item_embed_2, invariant_representation_as_query_2,
                                                        weight_matrix_2_1, weight_matrix_2_2)

        with tf.name_scope('encoder_1'):
            encoder_output_1, encoder_state_1 = self.GRU_Model_1(input_item_embed_1, each_step_cross_embed_1,
                                                                 self.len_1, self.hidden_size, self.num_layers,
                                                                 self.keep_prob)
        with tf.name_scope('encoder_2'):
            encoder_output_2, encoder_state_2 = self.GRU_Model_2(input_item_embed_2, each_step_cross_embed_2,
                                                                 self.len_2, self.hidden_size, self.num_layers,
                                                                 self.keep_prob)

        in_dim = self.embedding_size * 3
        out_dim = self.hidden_size
        sequential_aggregation_matrix_1_1 = tf.get_variable('sequential_aggregation_matrix_1_1',
                                                            shape=[in_dim, out_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer(
                                                                uniform=False))
        sequential_aggregation_matrix_2_1 = tf.get_variable('sequential_aggregation_matrix_2_1',
                                                            shape=[in_dim, out_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer(
                                                                uniform=False))
        in_dim = self.hidden_size
        out_dim = 1
        sequential_aggregation_matrix_1_2 = tf.get_variable('sequential_aggregation_matrix_1_2',
                                                            shape=[in_dim, out_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer(
                                                                uniform=False))
        sequential_aggregation_matrix_2_2 = tf.get_variable('sequential_aggregation_matrix_2_2',
                                                            shape=[in_dim, out_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer(
                                                                uniform=False))

        sequence_representation_final_1 = self.sequential_att_aggregation(specific_user_1_for_1, encoder_output_1,
                                                                          sequential_aggregation_matrix_1_1,
                                                                          sequential_aggregation_matrix_1_2)
        sequence_representation_final_2 = self.sequential_att_aggregation(specific_user_2_for_2, encoder_output_2,
                                                                          sequential_aggregation_matrix_2_1,
                                                                          sequential_aggregation_matrix_2_2)

        if self.agg_mode == 'concat':
            user_interest_1 = tf.concat(
                [sequence_representation_final_1, specific_user_1_for_1, invariant_user_1_for_1],
                axis=-1)
            user_interest_2 = tf.concat(
                [sequence_representation_final_2, specific_user_2_for_2, invariant_user_2_for_2],
                axis=-1)
        elif self.agg_mode == 'sum':
            user_interest_1 = tf.reduce_sum(
                [sequence_representation_final_1, specific_user_1_for_1, invariant_user_1_for_1],
                axis=-1, keep_dims=True)
            user_interest_2 = tf.reduce_sum(
                [sequence_representation_final_2, specific_user_2_for_2, invariant_user_2_for_2],
                axis=-1, keep_dims=True)
        elif self.agg_mode == 'mean':
            user_interest_1 = tf.reduce_mean(
                [sequence_representation_final_1, specific_user_1_for_1, invariant_user_1_for_1],
                axis=-1, keep_dims=True)
            user_interest_2 = tf.reduce_mean(
                [sequence_representation_final_2, specific_user_2_for_2, invariant_user_2_for_2],
                axis=-1, keep_dims=True)
        elif self.agg_mode == 'max':
            user_interest_1 = tf.reduce_max(
                [sequence_representation_final_1, specific_user_1_for_1, invariant_user_1_for_1],
                axis=-1, keep_dims=True)
            user_interest_2 = tf.reduce_max(
                [sequence_representation_final_2, specific_user_2_for_2, invariant_user_2_for_2],
                axis=-1, keep_dims=True)

        concat_output_1 = tf.concat([user_interest_1, target_item_embed_1], axis=-1)
        concat_output_1 = tf.nn.dropout(concat_output_1, self.keep_prob)
        # concat_output=[batch_size,hidden_size+hidden_size+hidden_size]
        hidden_pred_1 = tf.layers.dense(concat_output_1, self.intra_dim, activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        y_hat_1 = tf.layers.dense(hidden_pred_1, 1, activation=None,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        concat_output_2 = tf.concat([user_interest_2, target_item_embed_2], axis=-1)
        concat_output_2 = tf.nn.dropout(concat_output_2, self.keep_prob)
        # concat_output=[batch_size,hidden_size+hidden_size+hidden_size]
        hidden_pred_2 = tf.layers.dense(concat_output_2, self.intra_dim, activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        y_hat_2 = tf.layers.dense(hidden_pred_2, 1, activation=None,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.pred_score_1 = tf.reshape(y_hat_1, [-1])
        self.pred_score_2 = tf.reshape(y_hat_2, [-1])

        self.loss_main_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(y_hat_1, [-1]),
                                                                                  labels=self.y_1))
        self.loss_main_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(y_hat_2, [-1]),
                                                                                  labels=self.y_2))

        self.ssl_loss_1 = self.calc_ssl_loss_strategy(invariant_user_1_for_1, invariant_user_2_for_1,
                                                      specific_user_1_for_1, specific_user_2_for_1)
        self.ssl_loss_2 = self.calc_ssl_loss_strategy(invariant_user_1_for_2, invariant_user_2_for_2,
                                                      specific_user_1_for_2, specific_user_2_for_2)
        self.loss_1 = self.loss_main_1 + self.ssl_loss_1
        self.loss_2 = self.loss_main_2 + self.ssl_loss_2

        self.opt_1 = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.updates_1 = self.opt_1.minimize(self.loss_1)

        self.opt_2 = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.updates_2 = self.opt_2.minimize(self.loss_2)

    def get_gru_cell(self, hidden_size, keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob,
                                                 state_keep_prob=keep_prob)
        return gru_cell

    def GRU_Model_1(self, input_item_embed, step_wise_embed, len_1, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_A'):
            encoder_cell_1 = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob)
                                                          for _ in range(num_layers)])
            encoder_output_1, encoder_state_1 = tf.nn.dynamic_rnn(encoder_cell_1,
                                                                  tf.concat([input_item_embed, step_wise_embed],
                                                                            axis=-1),
                                                                  sequence_length=len_1, dtype=tf.float32)
            # encoder_output_A=[batch_size,timestamp_A,hidden_size],
            # encoder_state_A=([batch_size,hidden_size]*num_layers)
        return encoder_output_1, encoder_state_1

    def GRU_Model_2(self, input_item_embed, step_wise_embed, len_2, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_B'):
            encoder_cell_2 = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob)
                                                          for _ in range(num_layers)])
            encoder_output_2, encoder_state_2 = tf.nn.dynamic_rnn(encoder_cell_2,
                                                                  tf.concat([input_item_embed, step_wise_embed],
                                                                            axis=-1),
                                                                  sequence_length=len_2, dtype=tf.float32)
            # encoder_output_B=[batch_size,timestamp_B,hidden_size],
            # encoder_state_B=([batch_size,hidden_size]*num_layers)
        return encoder_output_2, encoder_state_2

    def transfer_2_to_1(self, encoder_output, sequence_len, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('transfer_B'):
            transfer_cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob)
                                                         for _ in range(num_layers)])
            transfer_output, transfer_state = tf.nn.dynamic_rnn(transfer_cell, encoder_output,
                                                                sequence_length=sequence_len, dtype=tf.float32)
            # transfer_output=[batch_size,timestamp_B,hidden_size],
            # transfer_state=([batch_size,hidden_size]*num_layers)
        return transfer_output, transfer_state

    def transfer_1_to_2(self, encoder_output, sequence_len, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('transfer_A'):
            transfer_cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob)
                                                         for _ in range(num_layers)])
            transfer_output, transfer_state = tf.nn.dynamic_rnn(transfer_cell, encoder_output,
                                                                sequence_length=sequence_len,
                                                                dtype=tf.float32)
            # transfer_output_A=[batch_size,timestamp_A,hidden_size],
            # transfer_state_A=([batch_size,hidden_size]*num_layers)
        return transfer_output, transfer_state

    def generate_mask_item_embed(self, item_embed_matrix, interacted_item_index):
        data_mask = tf.cast(tf.greater(interacted_item_index, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis=3)
        data_mask = tf.tile(data_mask, (1, 1, 1, self.embedding_size))

        data_embed = tf.nn.embedding_lookup(item_embed_matrix, interacted_item_index)
        data_embed_masked = tf.multiply(data_embed, data_mask)
        return data_embed_masked

    def reshape_item_embed(self, item_embed, max_num_each_step, max_length):
        # [batch_size, 1, embedding_size]
        data_embed_exp = tf.expand_dims(item_embed, 1)
        # [batch_size, 1, 1, embedding_size]
        data_embed_exp = tf.expand_dims(data_embed_exp, 1)
        # [batch_size, max_length, max_num_each_step, embedding_size]
        data_embed_tile = tf.tile(data_embed_exp, [1, max_length, max_num_each_step, 1])
        # [batch_size*max_length*max_num_each_step, embedding_size]
        item_embed_reshape = tf.reshape(data_embed_tile, [-1, self.embedding_size])
        return item_embed_reshape

    def agg_by_attention(self, step_wise_cross_interacted_item, interacted_item, weight_matrix_1, weight_matrix_2):
        # [batch_size*max_length*each_step_max_length, embedding_size]
        step_wise_cross_interacted_item_reshape = tf.reshape(step_wise_cross_interacted_item, [-1, self.embedding_size])
        # [batch_size*max_length*each_step_max_length, embedding_size*3]
        cur_input = tf.concat([step_wise_cross_interacted_item_reshape, interacted_item,
                               tf.multiply(step_wise_cross_interacted_item_reshape, interacted_item)], 1)
        # [batch_size*max_length*each_step_max_length, hidden_size]
        hidden = tf.nn.relu(tf.matmul(cur_input, weight_matrix_1))
        # [batch_size*max_length*each_step_max_length, 1]
        weight = tf.matmul(hidden, weight_matrix_2)
        # [batch_size, max_length, each_step_max_length, 1]
        weight = tf.reshape(weight, [-1, self.max_interacted_item, self.max_length_each_step, 1])
        normalize_weight = tf.nn.softmax(weight, dim=2)
        data_embed_cross_item_ig = step_wise_cross_interacted_item * normalize_weight
        # [batch_size, max_length, embedding_size]
        data_embed_agg_item = tf.reduce_mean(data_embed_cross_item_ig, 2)
        return data_embed_agg_item

    def calc_ssl_loss_strategy(self, invariant_embed_1, invariant_embed_2, specific_embed_1, specific_embed_2):
        normalize_invariant_user_1 = tf.nn.l2_normalize(invariant_embed_1, 1)
        normalize_invariant_user_2 = tf.nn.l2_normalize(invariant_embed_2, 1)

        normalize_specific_user_1 = tf.nn.l2_normalize(specific_embed_1, 1)
        normalize_specific_user_2 = tf.nn.l2_normalize(specific_embed_2, 1)

        pos_score_user = tf.reduce_sum(tf.multiply(normalize_invariant_user_1, normalize_invariant_user_2), axis=1)

        neg_score_1 = tf.reduce_sum(tf.multiply(normalize_invariant_user_1, normalize_specific_user_1), axis=1)
        neg_score_2 = tf.reduce_sum(tf.multiply(normalize_invariant_user_2, normalize_specific_user_2), axis=1)
        neg_score_3 = tf.reduce_sum(tf.multiply(normalize_specific_user_1, normalize_specific_user_2), axis=1)
        # neg_score_4 = tf.matmul(normalize_specific_user_1, normalize_specific_user_1,
        #                         transpose_a=False, transpose_b=True)
        # neg_score_5 = tf.matmul(normalize_specific_user_2, normalize_specific_user_2,
        #                         transpose_a=False, transpose_b=True)
        neg_score_4 = tf.matmul(normalize_invariant_user_1, normalize_invariant_user_2, transpose_a=False,
                                transpose_b=True)

        pos_score = tf.exp(pos_score_user / self.ssl_temp)
        neg_score_1 = tf.exp(neg_score_1 / self.ssl_temp)
        neg_score_2 = tf.exp(neg_score_2 / self.ssl_temp)
        neg_score_3 = tf.exp(neg_score_3 / self.ssl_temp)
        neg_score_4 = tf.reduce_sum(tf.exp(neg_score_4 / self.ssl_temp), axis=1)
        # neg_score_4 = tf.reduce_sum(tf.exp(neg_score_4 / self.ssl_temp), axis=1)
        # neg_score_5 = tf.reduce_sum(tf.exp(neg_score_5 / self.ssl_temp), axis=1)

        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score / (neg_score_1 + neg_score_2 + neg_score_3 + pos_score +
                                                           neg_score_4)))
        ssl_loss = self.ssl_reg * ssl_loss_user
        return ssl_loss

    def sequential_att_aggregation(self, query, sequential_hidden_states, weight_matrix_1, weight_matrix_2):
        # query: [batch_size, embedding_size]
        # sequential_hidden_states: [batch_size,timestamp_A,embedding_size]
        # query_exp: [batch_size, 1, embedding_size]
        query_exp = tf.expand_dims(query, 1)
        # [batch_size, max_length, embedding_size]
        data_embed_tile = tf.tile(query_exp, [1, self.max_interacted_item, 1])
        # [batch_size*max_length, embedding_size]
        item_embed_reshape = tf.reshape(data_embed_tile, [-1, self.embedding_size])
        # [batch_size*max_length, embedding_size]
        sequential_hidden_states_reshape = tf.reshape(sequential_hidden_states, [-1, self.embedding_size])
        # [batch_size*max_length, embedding_size*3]
        cur_input = tf.concat([sequential_hidden_states_reshape, item_embed_reshape,
                               tf.multiply(sequential_hidden_states_reshape, item_embed_reshape)], 1)
        # [batch_size*max_length, hidden_size]
        hidden = tf.nn.relu(tf.matmul(cur_input, weight_matrix_1))
        # [batch_size*max_length, 1]
        weight = tf.matmul(hidden, weight_matrix_2)
        # [batch_size, max_length, 1]
        weight = tf.reshape(weight, [-1, self.max_interacted_item, 1])
        normalize_weight = tf.nn.softmax(weight, dim=1)
        data_embed_cross_item_ig = sequential_hidden_states * normalize_weight
        # [batch_size, embedding_size]
        data_embed_agg_item = tf.reduce_mean(data_embed_cross_item_ig, 1)
        return data_embed_agg_item
