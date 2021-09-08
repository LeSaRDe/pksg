import sys
g_pksg_module_folder = '/home/mf3jh/workspace/pksg/'
sys.path.insert(1, g_pksg_module_folder)
import global_settings


# g_raw_ds_names = {'ca': 'CA_Mask_Twt_Text_Meta.csv',
#                   'ny': 'NY_Mask_Twt_Text_Meta.csv',
#                   'fl': 'FL_Mask_Twt_Text_Meta.csv',
#                   'pa': 'PA_Mask_Twt_Text_Meta.csv'}

# g_raw_ds_names = ['cp4.venezuela.twitter.training.anon.v3.2018-12-24_2018-12-31.json',
#                   'cp4.venezuela.twitter.training.anon.v3.2019-01-01_2019-01-14.json',
#                   'cp4.venezuela.twitter.training.anon.v3.2019-01-15_2019-01-20.json',
#                   'cp4.venezuela.twitter.training.anon.v3.2019-01-21_2019-01-24.json',
#                   'cp4.venezuela.twitter.training.anon.v3.2019-01-25_2019-01-31.json',
#                   'collection1_2019-02-01_2019-02-07_twitter_raw_data.json',
#                   'collection1_2019-02-08_2019-02-14_twitter_raw_data.json',
#                   'cp4.ven.ci2.twitter.v2.2019-02-15_2019-02-21.json',
#                   'cp4.ven.ci2.twitter.v2.2019-02-22_2019-02-28.json',
#                   'cp4.ven.ci2.twitter.v2.2019-03-01_2019-03-07.json',
#                   'cp4.ven.ci2.twitter.v2.2019-03-08_2019-03-14.json',
#                   'cp4.ven.ci2.twitter.v2.2019-03-15_2019-03-21.json',
#                   'cp4.ven.ci2.twitter.v2.2019-03-22_2019-04-04.json']

# g_raw_ds_names = ['my_test.txt']

# g_raw_ds_names = ['Restaurants_Train_v2.xml',
#                   'Laptop_Train_v2.xml']

g_raw_ds_names = ['sample.txt']


# Used by SEM_UNIT
g_cust_ph_file_fmt = global_settings.g_conf_folder + 'cust_ph_{0}.txt'

# Used by SENT_TS
g_query_file_fmt = global_settings.g_conf_folder + 'query_{0}.txt'