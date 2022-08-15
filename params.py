
N_EPOCH = 100
BATCH_SIZE = 500
LR = 0.001
MAX_NUM_ITEM = 20
Embedding_Size = 64
inter_dim = 64
item_att_hidden_dim = 64
interest_att_hidden_dim = 64
layer_dim_1 = [64, 32, 16, 1]
layer_dim_2 = [64, 32, 16, 1]

metaName_2 = 'CDs_and_Vinyl'
metaName_1 = 'Books'
# metaName_2 = 'Movies_and_TV'

# metaName_1 = 'Electronics'
# metaName_2 = 'Clothing_Shoes_and_Jewelry'
#
# metaName_1 = 'Amazon_Instant_Video'
# metaName_2 = 'Musical_Instruments'

home_path = '/home/zangtianzi/self_Supervised_CDR/Data/'+metaName_1+'_'+metaName_2+'/'
# home_path = '/Users/zangtianzi/PycharmProjects/CDSR_baseline/Data/'
#
# train_test_path = '/Users/zangtianzi/PycharmProjects/CDSR_baseline/Data/test_reserve_1/'
train_test_path = '/home/zangtianzi/self_Supervised_CDR/Data/test_reserve_1/'+metaName_1+'_'+metaName_2+'/'

overlapping_users_all_item_list_file_1 = home_path + metaName_1 + '_overlapping_users_all_item_list.dat'
overlapping_users_all_item_list_file_2 = home_path + metaName_2 + '_overlapping_users_all_item_list.dat'

# for PiNet
hidden_size = 64
intra_dim = 32
num_layers = 1
