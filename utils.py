import random
import numpy as np
import pickle
from itertools import islice
import params

batch_size = params.BATCH_SIZE


def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = i
        idx2term[i] = terms[i]
    return term2idx, idx2term


# class DataInput:
#     def __init__(self, data, batch_size):
#         self.batch_size = batch_size
#         self.data = data
#         self.epoch_size = len(self.data) // self.batch_size
#         #
#         self.i = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.i == self.epoch_size:
#             raise StopIteration
#         ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
#         self.i += 1
#         u, hist, hist_cross, i, y, sequence_length, sequence_length_cross, data_augmented = [], [], [], [], [], [], [], []
#         for t in ts:
#             u.append(t[0])
#             hist.append(t[1])
#             hist_cross.append(t[2])
#             i.append(t[3])
#             y.append(t[4])
#             sequence_length.append(t[5])
#             sequence_length_cross.append(t[6])
#             data_augmented.append(t[7])
#         return u, hist, hist_cross, i, y, sequence_length, sequence_length_cross, data_augmented

class DataInput_v3:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        #
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, y, sequence_length, sequence_length_cross, augmentation_seq_1, augmentation_seq_2 = \
            [], [], [], [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            y.append(t[4])
            sequence_length.append(t[5])
            sequence_length_cross.append(t[6])
            augmentation_seq_1.append(t[7])
            augmentation_seq_2.append(t[8])
        return u, hist, hist_cross, i, y, sequence_length, sequence_length_cross, augmentation_seq_1, augmentation_seq_2


class DataInput_v6:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        #
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, y, sequence_length, all_history_cross, sequence_length_cross = \
            [], [], [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            y.append(t[4])
            sequence_length.append(t[5])
            all_history_cross.append(t[6])
            sequence_length_cross.append(t[7])
        return u, hist, hist_cross, i, y, sequence_length, all_history_cross, sequence_length_cross


class DataInput_v2:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        #
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, y, sequence_length, cross_sequence_length = [], [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            y.append(t[4])
            sequence_length.append(t[5])
            cross_sequence_length.append(t[6])
        return u, hist, hist_cross, i, y, sequence_length, cross_sequence_length



class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        #
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, y, sequence_length = [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            y.append(t[4])
            sequence_length.append(t[5])
        return u, hist, hist_cross, i, y, sequence_length


def compute_auc(sess, model, testset1, testset2):
    arr_1, arr_2 = [], []
    for uij_1, uij_2 in zip(DataInput(testset1, batch_size), DataInput(testset2, batch_size)):
        a, b = model.test(sess, uij_1, uij_2)
        score, label, user = a
        # print(score)
        for index in range(len(score)):
            if label[index] > 0:
                arr_1.append([0, 1, score[index]])
            elif label[index] == 0:
                arr_1.append([1, 0, score[index]])
        score, label, user = b
        for index in range(len(score)):
            if label[index] > 0:
                arr_2.append([0, 1, score[index]])
            elif label[index] == 0:
                arr_2.append([1, 0, score[index]])
    arr_1 = sorted(arr_1, key=lambda d: d[2])
    arr_2 = sorted(arr_2, key=lambda d: d[2])
    auc_1 = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr_1:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc_1 += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_1) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc_1 = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc_1 = (1.0 - auc_1 / (2.0 * tp2 * fp2))
    else:
        auc_1 = None

    auc_2 = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr_2:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc_2 += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_2) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc_2 = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc_2 = (1.0 - auc_2 / (2.0 * tp2 * fp2))
    else:
        auc_2 = -0.5

    return auc_1, auc_2


def compute_hr(sess, model, testset1, testset2):
    hit_1, arr_1, hit_2, arr_2 = [], [], [], []
    userid1 = list(set([x[0] for x in testset1]))
    userid2 = list(set([x[0] for x in testset2]))
    for uij_1, uij_2 in zip(DataInput(testset1, batch_size), DataInput(testset2, batch_size)):
        a, b = model.test(sess, uij_1, uij_2)
        score, label, user = a
        for index in range(len(score)):
            if score[index] > 0.5:
                arr_1.append([label[index], 1, user[index]])
            else:
                arr_1.append([label[index], 0, user[index]])
        score, label, user = b
        for index in range(len(score)):
            if score[index] > 0.5:
                arr_2.append([label[index], 1, user[index]])
            else:
                arr_2.append([label[index], 0, user[index]])
    for user in userid1:
        arr_user = [x for x in arr_1 if x[2] == user and x[1] == 1]
        if len(arr_user) > 0:
            hit_1.append(sum([x[0] for x in arr_user]) / len(arr_user))
    for user in userid2:
        arr_user = [x for x in arr_2 if x[2] == user and x[1] == 1]
        if len(arr_user) > 0:
            hit_2.append(sum([x[0] for x in arr_user]) / len(arr_user))
    return np.mean(hit_1), np.mean(hit_2)


def load_data(trainset_file, testset_file):
    with open(trainset_file, 'rb') as train_file:
        trainset = pickle.load(train_file)

    with open(testset_file, 'rb') as test_file:
        testset = pickle.load(test_file)
    user_set = set()
    item_set = set()
    for i in range(len(trainset)):
        user_set.add(trainset[i][0])
        for item in trainset[i][1]:
            item_set.add(item)
        item_set.add(trainset[i][3])
    for i in range(len(testset)):
        user_set.add(testset[i][0])
        for item in testset[i][1]:
            item_set.add(item)
        item_set.add(testset[i][3])
    return trainset, testset, len(user_set), len(item_set)


def generate_all_item_dict(all_item_list_file):
    all_item_list = open(all_item_list_file)
    domain_all_item_dict = {}
    for line in islice(all_item_list, 0, None):
        u = line.split(' ')[0]
        items = line.strip('\n').split(' ')[1:]
        items = [int(x) for x in items]
        domain_all_item_dict[u] = items
    return domain_all_item_dict


def generate_train_sample_for_each_epoch(trainset, all_item_dict, number_of_item):
    for instance_index in range(len(trainset)):
        u = trainset[instance_index][0]
        history_interacted_item = trainset[instance_index][1]
        history_cross_interacted_item = trainset[instance_index][2]
        sequence_length = trainset[instance_index][5]
        sequence_length_cross = trainset[instance_index][6]
        negative_sample = random.randint(0, number_of_item - 1)
        while negative_sample in all_item_dict[str(u)]:
            negative_sample = random.randint(0, number_of_item - 1)
        ground_truth = 0.0
        trainset.append((u, history_interacted_item, history_cross_interacted_item, negative_sample, ground_truth,
                         sequence_length, sequence_length_cross))
    return trainset, len(trainset)


def generate_test_instance(testset1, all_item_dict_1, number_of_item_1):
    for sample_index in range(len(testset1)):
        user_instance = []
        history_interacted_item_instance = []
        history_cross_interacted_item_instance = []
        target_item_instance = []
        ground_truth_instance = []
        sequence_length_instance = []
        sequence_length_cross_instance = []

        u = testset1[sample_index][0]
        history_interacted_item = testset1[sample_index][1]
        history_cross_interacted_item = testset1[sample_index][2]
        target_item = testset1[sample_index][3]
        ground_truth = testset1[sample_index][4]
        sequence_length = testset1[sample_index][5]
        sequence_length_cross = testset1[sample_index][6]

        user_instance.append(u)
        history_interacted_item_instance.append(history_interacted_item)
        history_cross_interacted_item_instance.append(history_cross_interacted_item)
        target_item_instance.append(target_item)
        ground_truth_instance.append(ground_truth)
        sequence_length_instance.append(sequence_length)
        sequence_length_cross_instance.append(sequence_length_cross)

        ground_truth = 0.0
        for j in range(99):
            k = np.random.randint(0, number_of_item_1-1)
            while k in all_item_dict_1[str(u)]:
                k = np.random.randint(0, number_of_item_1-1)
            user_instance.append(u)
            history_interacted_item_instance.append(history_interacted_item)
            history_cross_interacted_item_instance.append(history_cross_interacted_item)
            target_item_instance.append(k)
            ground_truth_instance.append(ground_truth)
            sequence_length_instance.append(sequence_length)
            sequence_length_cross_instance.append(sequence_length_cross)
        generated_test_instance = [user_instance, history_interacted_item_instance,
                                   history_cross_interacted_item_instance, target_item_instance, ground_truth_instance,
                                   sequence_length_instance, sequence_length_cross_instance]
        yield generated_test_instance


def generate_test_instance_v4(testset1, all_item_dict_1, number_of_item_1):
    for sample_index in range(len(testset1)):
        user_instance = []
        history_interacted_item_instance = []
        history_cross_interacted_item_instance = []
        target_item_instance = []
        ground_truth_instance = []
        sequence_length_instance = []

        u = testset1[sample_index][0]
        history_interacted_item = testset1[sample_index][1]
        history_cross_interacted_item = testset1[sample_index][2]
        target_item = testset1[sample_index][3]
        ground_truth = testset1[sample_index][4]
        sequence_length = testset1[sample_index][5]

        user_instance.append(u)
        history_interacted_item_instance.append(history_interacted_item)
        history_cross_interacted_item_instance.append(history_cross_interacted_item)
        target_item_instance.append(target_item)
        ground_truth_instance.append(ground_truth)
        sequence_length_instance.append(sequence_length)

        ground_truth = 0.0
        for j in range(99):
            k = np.random.randint(0, number_of_item_1-1)
            while k in all_item_dict_1[str(u)]:
                k = np.random.randint(0, number_of_item_1-1)
            user_instance.append(u)
            history_interacted_item_instance.append(history_interacted_item)
            history_cross_interacted_item_instance.append(history_cross_interacted_item)
            target_item_instance.append(k)
            ground_truth_instance.append(ground_truth)
            sequence_length_instance.append(sequence_length)
        generated_test_instance = [user_instance, history_interacted_item_instance,
                                   history_cross_interacted_item_instance, target_item_instance, ground_truth_instance,
                                   sequence_length_instance]
        yield generated_test_instance


def generate_test_instance_v6(testset1, all_item_dict_1, number_of_item_1):
    for sample_index in range(len(testset1)):
        user_instance = []
        history_interacted_item_instance = []
        history_cross_interacted_item_instance = []
        target_item_instance = []
        ground_truth_instance = []
        sequence_length_instance = []
        all_history_cross_interacted_item_instance = []
        sequence_length_cross_instance = []

        u = testset1[sample_index][0]
        history_interacted_item = testset1[sample_index][1]
        history_cross_interacted_item = testset1[sample_index][2]
        target_item = testset1[sample_index][3]
        ground_truth = testset1[sample_index][4]
        sequence_length = testset1[sample_index][5]
        all_history_cross_interacted_item = testset1[sample_index][6]
        sequence_length_cross = testset1[sample_index][7]

        user_instance.append(u)
        history_interacted_item_instance.append(history_interacted_item)
        history_cross_interacted_item_instance.append(history_cross_interacted_item)
        target_item_instance.append(target_item)
        ground_truth_instance.append(ground_truth)
        sequence_length_instance.append(sequence_length)
        all_history_cross_interacted_item_instance.append(all_history_cross_interacted_item)
        sequence_length_cross_instance.append(sequence_length_cross)

        ground_truth = 0.0
        for j in range(99):
            k = np.random.randint(0, number_of_item_1-1)
            while k in all_item_dict_1[str(u)]:
                k = np.random.randint(0, number_of_item_1-1)
            user_instance.append(u)
            history_interacted_item_instance.append(history_interacted_item)
            history_cross_interacted_item_instance.append(history_cross_interacted_item)
            target_item_instance.append(k)
            ground_truth_instance.append(ground_truth)
            sequence_length_instance.append(sequence_length)
            all_history_cross_interacted_item_instance.append(all_history_cross_interacted_item)
            sequence_length_cross_instance.append(sequence_length_cross)
        generated_test_instance = [user_instance, history_interacted_item_instance,
                                   history_cross_interacted_item_instance, target_item_instance, ground_truth_instance,
                                   sequence_length_instance, all_history_cross_interacted_item_instance,
                                   sequence_length_cross_instance]
        yield generated_test_instance


def sequence_augmentation(sequence, sequence_length):
    random_num = int(sequence_length * 0.2)
    random_position = random.sample(list(range(sequence_length)), random_num)
    for position in random_position:
        sequence[position] = int(0)
    return sequence


def generate_train_sample_for_each_epoch_v3(trainset, all_item_dict, number_of_item):
    for instance_index in range(len(trainset)):

        u = trainset[instance_index][0]
        history_interacted_item = trainset[instance_index][1]
        history_cross_interacted_item = trainset[instance_index][2]
        sequence_length = trainset[instance_index][5]
        sequence_length_cross = trainset[instance_index][6]
        augmentation_sequence_1 = sequence_augmentation(history_interacted_item[:], sequence_length)
        trainset[instance_index].append(augmentation_sequence_1)
        augmentation_sequence_2 = sequence_augmentation(history_interacted_item[:], sequence_length)
        trainset[instance_index].append(augmentation_sequence_2)

        negative_sample = random.randint(0, number_of_item - 1)
        while negative_sample in all_item_dict[str(u)]:
            negative_sample = random.randint(0, number_of_item - 1)
        ground_truth = 0.0
        trainset.append([u, history_interacted_item, history_cross_interacted_item, negative_sample, ground_truth,
                         sequence_length, sequence_length_cross, augmentation_sequence_1, augmentation_sequence_2])
    return trainset, len(trainset)


def generate_train_sample_for_each_epoch_v4(trainset, all_item_dict, number_of_item):
    for instance_index in range(len(trainset)):
        u = trainset[instance_index][0]
        history_interacted_item = trainset[instance_index][1]
        history_cross_interacted_item = trainset[instance_index][2]
        sequence_length = trainset[instance_index][5]

        augmentation_sequence = sequence_augmentation(history_interacted_item[:], sequence_length)
        trainset[instance_index].append(augmentation_sequence)

        negative_sample = random.randint(0, number_of_item - 1)
        while negative_sample in all_item_dict[str(u)]:
            negative_sample = random.randint(0, number_of_item - 1)
        ground_truth = 0.0
        trainset.append([u, history_interacted_item, history_cross_interacted_item, negative_sample, ground_truth,
                         sequence_length, augmentation_sequence])
    return trainset, len(trainset)


def generate_train_sample_for_each_epoch_v6(trainset, all_item_dict, number_of_item):
    for instance_index in range(len(trainset)):
        u = trainset[instance_index][0]
        history_interacted_item = trainset[instance_index][1]
        history_cross_interacted_item = trainset[instance_index][2]
        sequence_length = trainset[instance_index][5]
        all_history_cross_interacted_item = trainset[instance_index][6]
        sequence_length_cross = trainset[instance_index][7]

        negative_sample = random.randint(0, number_of_item - 1)
        while negative_sample in all_item_dict[str(u)]:
            negative_sample = random.randint(0, number_of_item - 1)
        ground_truth = 0.0
        trainset.append([u, history_interacted_item, history_cross_interacted_item, negative_sample, ground_truth,
                         sequence_length, all_history_cross_interacted_item, sequence_length_cross])
    return trainset, len(trainset)