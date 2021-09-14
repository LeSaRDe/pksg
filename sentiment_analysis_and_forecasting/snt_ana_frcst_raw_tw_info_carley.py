import logging
import time
import sys
import pandas as pd
import csv

import scenario_settings
from lib import global_settings

"""
PKSG PIPELINE
STAGE: RAW TEXT INFORMATION EXTRACTION (RAW_TXT)

REQUIREMENTS:
    Raw input data

OUTPUTS:
    text->raw text info table: g_raw_tw_info_file_fmt
        pandas DataFrame
        Index: tweet id (str): 'tw_id'
        Columns: as follows
        
        Given a set of raw tweet objects, extract the following fields:
        ['tw_id', 
        'tw_type', 
        'tw_datetime', 
        'tw_lang', 
        'tw_usr_id',
        'tw_src_id', 
        'tw_src_usr_id', 
        'tw_raw_txt', 
        'full_txt_flag',
        'tw_is_bot', 
        'tw_geo_state', 
        'tw_geo_city']
        
        'tw_type' does not have 'r';
        'tw_lang' is always 'en';
        'tw_src_id' is always 'None';
        'tw_src_usr_id' is always 'None';
        'full_txt_flag' is always 'True'

NOTE:
    This file is always scenario-specific.
"""


def get_tw_datetime(tw_date_str, tw_time_str):
    """
    :param tw_date_str: 'MM/DD/YYYY'
    :param tw_time_str: 'hh:mm:ss'
    :return: 'YYYYMMDDhhmmss'
    """
    l_date_fields = [item.strip() for item in tw_date_str.split('/')]
    l_time_fields = [item.strip() for item in tw_time_str.split(':')]
    return l_date_fields[2] + l_date_fields[0] + l_date_fields[1] + \
           l_time_fields[0] + l_time_fields[1] + l_time_fields[2]


def get_tw_type(is_quote_str, is_retweet_str):
    if is_quote_str != '1':
        is_quote = False
    else:
        is_quote = True
    if is_retweet_str != '1':
        is_retweet = False
    else:
        is_retweet = True

    if is_retweet:
        return 't'
    elif is_quote:
        return 'q'
    else:
        return 'n'


def get_tw_is_bot(is_bot_str):
    if is_bot_str != '1':
        return False
    else:
        return True


def extract_raw_tw_info_tasks(job_id, ds_name):
    '''
    Task ids are of the format: job_id#x
    where x ranges in [0, num_proc-1]
    '''
    logging.debug('[extract_raw_tw_info_tasks] Job %s: Starts.' % str(job_id))
    timer_start = time.time()

    if scenario_settings.g_raw_ds_names is None or len(scenario_settings.g_raw_ds_names) <= 0:
        raise Exception('[extract_raw_tw_info_tasks] Job %s: No raw dataset is found!' % str(job_id))

    l_tw_rec = []
    raw_ds_name = scenario_settings.g_raw_ds_names[str(ds_name)]
    logging.debug('[extract_raw_tw_info_tasks] Job %s: Loading %s ' % (job_id, raw_ds_name))
    with open(global_settings.g_raw_data_folder + raw_ds_name, 'r') as in_fd:
        csv_reader = csv.reader(in_fd, delimiter=',')
        ln_cnt = 0
        for row in csv_reader:
            if ln_cnt == 0:
                ln_cnt += 1
            else:
                tw_id = row[0]
                tw_usr_id = row[1]
                tw_geo_state = row[2]
                tw_date = row[3]
                tw_time = row[4]
                tw_raw_txt = row[7]
                tw_is_bot = get_tw_is_bot(row[8])
                tw_geo_city = row[9]
                tw_is_quote = row[10]
                tw_is_retweet = row[11]

                tw_datetime = get_tw_datetime(tw_date, tw_time)
                tw_type = get_tw_type(tw_is_quote, tw_is_retweet)
                tw_lang = 'en'
                tw_src_id = None
                tw_src_usr_id = None
                full_txt_flag = True
                l_tw_rec.append((tw_id, tw_type, tw_datetime, tw_lang, tw_usr_id, tw_src_id, tw_src_usr_id, tw_raw_txt,
                                 full_txt_flag, tw_is_bot, tw_geo_state, tw_geo_city))
                ln_cnt += 1
                if ln_cnt % 5000 == 0 and ln_cnt >= 5000:
                    logging.debug('[extract_raw_tw_info_tasks] Job %s: Scanned %s raw tws in %s secs.'
                                  % (job_id, ln_cnt, time.time() - timer_start))
        in_fd.close()
        logging.debug('[extract_raw_tw_info_tasks] Job %s: Scanned all %s raw tws in %s secs.'
                      % (job_id, ln_cnt, time.time() - timer_start))
    df_raw_tw_info = pd.DataFrame(l_tw_rec, columns=['tw_id', 'tw_type', 'tw_datetime', 'tw_lang', 'tw_usr_id',
                                                     'tw_src_id', 'tw_src_usr_id', 'tw_raw_txt', 'full_txt_flag',
                                                     'tw_is_bot', 'tw_geo_state', 'tw_geo_city'])
    df_raw_tw_info = df_raw_tw_info.set_index('tw_id')
    df_raw_tw_info = df_raw_tw_info[~df_raw_tw_info.index.duplicated(keep='first')]
    # output_ds_name = global_settings.g_ds_ids[int(job_id)]
    pd.to_pickle(df_raw_tw_info, global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[extract_raw_tw_info_tasks] Job %s: All done with %s raw_tw_info recs in %s secs.'
                  % (job_id, len(df_raw_tw_info), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'raw_tw_info':
        # raw_tw_info 0 ca
        job_id = sys.argv[2]
        ds_name = sys.argv[3]
        extract_raw_tw_info_tasks(job_id, ds_name)
    # elif cmd == 'raw_tw_info':
    #     # raw_tw_info 1 0 202001
    #     num_proc = sys.argv[2]
    #     job_id = sys.argv[3]
    #     ds_name = sys.argv[4]
    #     index_col = 'tw_id'
    #     extract_raw_tw_info_wrapper(num_proc, job_id, ds_name, index_col)
