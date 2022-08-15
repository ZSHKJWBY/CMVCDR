import os
import sys
import time
import random as rd
import numpy as np
import tensorflow as tf
from model import OurNet
from utils import *
from utils import DataInput, compute_auc, compute_hr, dictionary
import params
from progressbar import *


def best_result(best, current):
    # print("find the best number:")
    num_ret = len(best)
    for numIdx in range(num_ret):
        if float(current[numIdx]) > float(best[numIdx]):
            best[numIdx] = float(current[numIdx])
    return best


if __name__ == '__main__':
    print('begin to bulid Our model')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trainset1, testset1, number_of_user_1, number_of_item_1 = \
        load_data(trainset_file='data/'+params.metaName_1+'_'+params.metaName_2+'/' + params.metaName_1 + '_trainset.pickle',
                  testset_file='data/'+params.metaName_1+'_'+params.metaName_2+'/' + params.metaName_1 + '_testset.pickle')
    trainset2, testset2, number_of_user_2, number_of_item_2 = \
        load_data(trainset_file='data/'+params.metaName_1+'_'+params.metaName_2+'/' + params.metaName_2 + '_trainset.pickle',
                  testset_file='data/'+params.metaName_1+'_'+params.metaName_2+'/' + params.metaName_2 + '_testset.pickle')
    print(number_of_user_1, number_of_item_1)
    print(number_of_user_2, number_of_item_2)

    all_item_dict_1 = generate_all_item_dict(params.overlapping_users_all_item_list_file_1)
    all_item_dict_2 = generate_all_item_dict(params.overlapping_users_all_item_list_file_2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    best_ret_1 = np.array([0] * 6)
    best_ret_2 = np.array([0] * 6)

    with tf.Session(config=config) as sess:
        model = OurNet(number_of_user_1, number_of_item_1, number_of_item_2, params.LR, params.Embedding_Size,
                       params.hidden_size, params.intra_dim, params.num_layers)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        for epoch in range(params.N_EPOCH):
            start_time = time.time()
            ret_1 = np.array([0.0] * 6)
            ret_2 = np.array([0.0] * 6)

            print(epoch)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            generated_sample_1, num_of_sample_1 = generate_train_sample_for_each_epoch_v4(trainset1[:],
                                                                                          all_item_dict_1,
                                                                                          number_of_item_1)
            generated_sample_2, num_of_sample_2 = generate_train_sample_for_each_epoch_v4(trainset2[:],
                                                                                          all_item_dict_2,
                                                                                          number_of_item_2)

            rd.shuffle(generated_sample_1)
            rd.shuffle(generated_sample_2)

            loss_sum = 0.0
            loss_main_sum = 0.0
            loss_ss_sum = 0.0

            num_batches_1 = int(num_of_sample_1 / params.BATCH_SIZE)
            widgets_1 = ['Train1: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
            pbar_1 = ProgressBar(widgets=widgets_1, maxval=num_batches_1-1).start()
            i_1 = 0

            for uij in DataInput(generated_sample_1, params.BATCH_SIZE):
                pbar_1.update(i_1)
                feed_dict = {
                    model.u_1: uij[0],
                    model.input_1: uij[1],
                    model.hist_cross_item_1: uij[2],
                    model.target_1: uij[3],
                    model.y_1: uij[4],
                    model.len_1: uij[5],
                    model.keep_prob: 0.8,
                }
                i_1 += 1
                loss, _, loss_main, loss_ss = sess.run([model.loss_1, model.updates_1, model.loss_main_1,
                                                        model.ssl_loss_1], feed_dict=feed_dict)
                loss_sum += loss
                loss_main_sum += loss_main
                loss_ss_sum += loss_ss

            num_batches_2 = int(num_of_sample_2 / params.BATCH_SIZE)
            widgets_2 = ['Train2: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
            pbar_2 = ProgressBar(widgets=widgets_2, maxval=num_batches_2-1).start()
            i_2 = 0

            for uij in DataInput(generated_sample_2, params.BATCH_SIZE):
                pbar_2.update(i_2)
                feed_dict = {
                    model.u_2: uij[0],
                    model.input_2: uij[1],
                    model.hist_cross_item_2: uij[2],
                    model.target_2: uij[3],
                    model.y_2: uij[4],
                    model.len_2: uij[5],
                    model.keep_prob: 0.8,
                }
                i_2 += 1
                loss, _, loss_main, loss_ss = sess.run([model.loss_2, model.updates_2, model.loss_main_2,
                                                        model.ssl_loss_2], feed_dict=feed_dict)
                loss_sum += loss
                loss_main_sum += loss_main
                loss_ss_sum += loss_ss

            print('Epoch {}/{} - Training Loss: {:.3f} {:.3f} {:.3f}'.format(epoch + 1, params.N_EPOCH, loss_sum,
                                                                             loss_main_sum, loss_ss_sum))

            # 在两个domain上面分别做test，验证模型表现
            # domain1
            instance_count_1 = 0
            instance_count_2 = 0

            num_batches_3 = len(testset1)
            widgets_3 = ['Test1: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
            pbar_3 = ProgressBar(widgets=widgets_3, maxval=num_batches_3).start()
            i_3 = 0

            for test_instances_1 in generate_test_instance_v4(testset1[:], all_item_dict_1, number_of_item_1):
                pbar_3.update(i_3)
                score_1 = sess.run([model.pred_score_1], feed_dict={
                    model.u_1: test_instances_1[0],
                    model.input_1: test_instances_1[1],
                    model.hist_cross_item_1: test_instances_1[2],
                    model.target_1: test_instances_1[3],
                    model.y_1: test_instances_1[4],
                    model.len_1: test_instances_1[5],
                    model.keep_prob: 1.0
                })
                i_3 += 1
                instance_count_1 += 1
                predictions_1 = [-1 * i for i in score_1]
                rank_1 = np.array(predictions_1).argsort().argsort()[0][0]

                if rank_1 < 5:
                    ret_1[0] += 1
                    ret_1[3] += 1 / np.log2(rank_1 + 2)
                if rank_1 < 10:
                    ret_1[1] += 1
                    ret_1[4] += 1 / np.log2(rank_1 + 2)
                if rank_1 < 20:
                    ret_1[2] += 1
                    ret_1[5] += 1 / np.log2(rank_1 + 2)

            num_batches_4 = len(testset2)
            widgets_4 = ['Test2: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
            pbar_4 = ProgressBar(widgets=widgets_4, maxval=num_batches_3).start()
            i_4 = 0

            for test_instances_2 in generate_test_instance_v4(testset2[:], all_item_dict_2, number_of_item_2):
                pbar_4.update(i_4)
                score_2 = sess.run([model.pred_score_2], feed_dict={
                    model.u_2: test_instances_2[0],
                    model.input_2: test_instances_2[1],
                    model.hist_cross_item_2: test_instances_2[2],
                    model.target_2: test_instances_2[3],
                    model.y_2: test_instances_2[4],
                    model.len_2: test_instances_2[5],
                    model.keep_prob: 1.0
                })
                i_4 += 1
                instance_count_2 += 1
                predictions_2 = [-1 * i for i in score_2]

                rank_2 = np.array(predictions_2).argsort().argsort()[0][0]
                if rank_2 < 5:
                    ret_2[0] += 1
                    ret_2[3] += 1 / np.log2(rank_2 + 2)
                if rank_2 < 10:
                    ret_2[1] += 1
                    ret_2[4] += 1 / np.log2(rank_2 + 2)
                if rank_2 < 20:
                    ret_2[2] += 1
                    ret_2[5] += 1 / np.log2(rank_2 + 2)

            best_ret_2 = best_result(best_ret_2, ret_2)
            best_ret_1 = best_result(best_ret_1, ret_1)

            save_name = '/model_Ours/'

            print('%s: HR_5 %f HR_10 %f HR_20 %f'
                  % (params.metaName_1, ret_1[0] / instance_count_1, ret_1[1] / instance_count_1,
                     ret_1[2] / instance_count_1))
            print('%s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                  % (params.metaName_1, ret_1[3] / instance_count_1, ret_1[4] / instance_count_1,
                     ret_1[5] / instance_count_1))

            print('Best Result for %s: HR_5 %f HR_10 %f HR_20 %f'
                  % (params.metaName_1, best_ret_1[0] / instance_count_1,
                     best_ret_1[1] / instance_count_1,
                     best_ret_1[2] / instance_count_1))
            print('Best Result for %s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                  % (params.metaName_1, best_ret_1[3] / instance_count_1,
                     best_ret_1[4] / instance_count_1,
                     best_ret_1[5] / instance_count_1))

            print('%s: HR_5 %f HR_10 %f HR_20 %f'
                  % (params.metaName_2, ret_2[0] / instance_count_2, ret_2[1] / instance_count_2,
                     ret_2[2] / instance_count_2))
            print('%s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                  % (params.metaName_2, ret_2[3] / instance_count_2, ret_2[4] / instance_count_2,
                     ret_2[5] / instance_count_2))

            print('Best Result for %s: HR_5 %f HR_10 %f HR_20 %f'
                  % (params.metaName_2, best_ret_2[0] / instance_count_2,
                     best_ret_2[1] / instance_count_2,
                     best_ret_2[2] / instance_count_2))
            print('Best Result for %s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                  % (params.metaName_2, best_ret_2[3] / instance_count_2,
                     best_ret_2[4] / instance_count_2,
                     best_ret_2[5] / instance_count_2))
