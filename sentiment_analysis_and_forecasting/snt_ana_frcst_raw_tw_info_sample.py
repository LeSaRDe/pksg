import logging
import re
from os import walk
import time
import json
import math
import sys
import pandas as pd

# import scenario_settings
from lib import global_settings
from lib.multitasking import multitasking


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
        'tw_place_id',
        'tw_place_type', 
        'tw_place_full_name', 
        'tw_place_country_code']
        'full_txt_flag'=True : 'tw_raw_txt' takes from 'text'
        'full_txt_flag'=False :  'tw_raw_txt' takes from 'full_text'

NOTE:
    This file is always scenario-specific.
"""


def get_tw_id(tw_json):
    return tw_json['id_str']


def get_tw_type(tw_json):
    if 'in_reply_to_status_id_str' in tw_json \
            and tw_json['in_reply_to_status_id_str'] != '' \
            and not tw_json['in_reply_to_status_id_str'] is None:
        t_type = 'r'
    elif 'retweeted_status' in tw_json and tw_json['retweeted_status'] is not None:
        t_type = 't'
    elif 'quoted_status' in tw_json and tw_json['quoted_status'] is not None:
        t_type = 'q'
    else:
        t_type = 'n'
    return t_type


def get_tw_lang(tw_json):
    return tw_json['lang']


def get_tw_usr_id(tw_json):
    if 'user' not in tw_json:
        return None
    return tw_json['user']['id_str']


def get_tw_src_id(tw_json, tw_type):
    src_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'r' and 'in_reply_to_status_id_str' in tw_json:
        src_id = tw_json['in_reply_to_status_id_str']
    elif tw_type == 'q' and 'quoted_status_id_str' in tw_json:
        src_id = tw_json['quoted_status_id_str']
    elif tw_type == 't' and 'retweeted_status' in tw_json and 'id_str' in tw_json['retweeted_status']:
        src_id = tw_json['retweeted_status']['id_str']
    return src_id


def get_tw_src_usr_id(tw_json, tw_type):
    src_usr_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'q' and 'quoted_status' in tw_json and 'user' in tw_json['quoted_status']:
        src_usr_id = tw_json['quoted_status']['user']['id_str']
    elif tw_type == 'r':
        src_usr_id = tw_json['in_reply_to_user_id_str']
    elif tw_type == 'q' and 'retweeted_status' in tw_json and 'user' in tw_json['retweeted_status']:
        src_usr_id = tw_json['retweeted_status']['user']['id_str']
    return src_usr_id


def translate_month(month_str):
    month = None
    if month_str == 'Jan':
        month = '01'
    elif month_str == 'Feb':
        month = '02'
    elif month_str == 'Mar':
        month = '03'
    elif month_str == 'Apr':
        month = '04'
    elif month_str == 'May':
        month = '05'
    elif month_str == 'Jun':
        month = '06'
    elif month_str == 'Jul':
        month = '07'
    elif month_str == 'Aug':
        month = '08'
    elif month_str == 'Sep':
        month = '09'
    elif month_str == 'Oct':
        month = '10'
    elif month_str == 'Nov':
        month = '11'
    elif month_str == 'Dec':
        month = '12'
    else:
        raise Exception('Wrong month exists! user_time = %s' % month_str)
    return month


def get_tw_datetime(tw_json):
    '''
    Converte the datetime in the raw tweet object to the formart: YYYYMMDDHHMMSS
    '''
    if 'created_at' not in tw_json:
        return None
    date_fields = [item.strip() for item in tw_json['created_at'].split(' ')]
    mon_str = translate_month(date_fields[1])
    day_str = date_fields[2]
    year_str = date_fields[5]
    time_str = ''.join([item.strip() for item in date_fields[3].split(':')])
    return year_str + mon_str + day_str + time_str


def get_tw_raw_txt(tw_json, tw_type, tw_lang):
    '''
    Return (text, True/False for full_text)
    '''
    if tw_type == 'n' or tw_type == 'r' or tw_type == 'q':
        if tw_lang == 'en':
            if 'full_text' in tw_json:
                return (tw_json['full_text'], True)
            elif 'text' in tw_json:
                return (tw_json['text'], False)
            else:
                return None
    elif tw_type == 't':
        if tw_lang == 'en':
            if 'retweeted_status' in tw_json and 'full_text' in tw_json['retweeted_status']:
                return (tw_json['retweeted_status']['full_text'], True)
            elif 'retweeted_status' in tw_json and 'text' in tw_json['retweeted_status']:
                return (tw_json['retweeted_status']['text'], False)
            else:
                return None
    else:
        return None


def get_tw_geo(tw_json):
    if 'place' not in tw_json:
        return None
    if tw_json['place'] is None:
        return None
    place_id = tw_json['place']['id']
    place_type = tw_json['place']['place_type']
    place_full_name = tw_json['place']['full_name']
    place_country_code = tw_json['place']['country_code']

    return [place_id, place_type, place_full_name, place_country_code]


def extract_raw_tw_info_single_proc(task_id):
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    l_tw_json = []
    with open(global_settings.g_raw_txt_info_task_file_fmt.format(str(task_id)), 'r') as in_fd:
        for ln in in_fd:
            tw_str = ln.strip()
            tw_json = json.loads(tw_str)
            l_tw_json.append(tw_json)
        in_fd.close()
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: %s tw objs in total.' % (task_id, len(l_tw_json)))

    total_cnt = 0
    ready_cnt = 0
    l_ready = []
    for tw_json in l_tw_json:
        tw_id = get_tw_id(tw_json)
        tw_type = get_tw_type(tw_json)
        tw_datetime = get_tw_datetime(tw_json)
        tw_lang = get_tw_lang(tw_json)
        tw_usr_id = get_tw_usr_id(tw_json)
        tw_src_id = get_tw_src_id(tw_json, tw_type)
        tw_src_usr_id = get_tw_src_usr_id(tw_json, tw_type)
        tw_txt_ret = get_tw_raw_txt(tw_json, tw_type, tw_lang)
        tw_geo_info = get_tw_geo(tw_json)
        if tw_txt_ret is None:
            tw_raw_txt = None
            full_txt_flag = False
        else:
            tw_raw_txt = tw_txt_ret[0]
            full_txt_flag = tw_txt_ret[1]
        tw_place_id = None
        tw_place_type = None
        tw_place_full_name = None
        tw_place_country_code = None
        if tw_geo_info is not None:
            tw_place_id = tw_geo_info[0]
            tw_place_type = tw_geo_info[1]
            tw_place_full_name = tw_geo_info[2]
            tw_place_country_code = tw_geo_info[3]
        if tw_id is None or tw_type is None or tw_datetime is None or tw_usr_id is None:
            logging.error('[extract_raw_tw_info_single_proc] Proc %s: Incorrect tw json: %s' % (task_id, str(tw_json)))
        else:
            l_ready.append((tw_id, tw_type, tw_datetime, tw_lang, tw_usr_id, tw_src_id, tw_src_usr_id,
                            tw_raw_txt, full_txt_flag, tw_place_id, tw_place_type, tw_place_full_name,
                            tw_place_country_code))
            ready_cnt += 1
        total_cnt += 1
        if total_cnt % 5000 == 0 and total_cnt >= 5000:
            logging.debug('[extract_raw_tw_info_single_proc] Proc %s: total_cnt = %s ready_cnt = %s done in %s secs.'
                          % (task_id, total_cnt, ready_cnt, time.time() - timer_start))
    df_ready = pd.DataFrame(l_ready, columns=['tw_id', 'tw_type', 'tw_datetime', 'tw_lang', 'tw_usr_id', 'tw_src_id',
                                              'tw_src_usr_id', 'tw_raw_txt', 'full_txt_flag', 'tw_place_id',
                                              'tw_place_type', 'tw_place_full_name', 'tw_place_country_code'])
    pd.to_pickle(df_ready, global_settings.g_raw_txt_info_int_file_fmt.format(task_id))
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: All done with %s recs in %s secs.'
                  % (task_id, len(df_ready), time.time() - timer_start))


def after_merge_processing(df_merge):
    # Remove duplicates.
    df_merge = df_merge[~df_merge.index.duplicated(keep='first')]

    # Simplify retweets by removing their texts if the texts have already exists in the original tweets.
    df_merge['rt_ref'] = None
    logging.debug('[after_merge_processing] Load in df_raw_tw_info with %s recs.' % str(len(df_merge)))

    timer_start = time.time()
    total_cnt = 0
    ref_cnt = 0
    for tw_id, raw_tw_info in df_merge.iterrows():
        total_cnt += 1
        if total_cnt % 5000 == 0 and total_cnt >= 5000:
            logging.debug('[after_merge_processing] %s tws scanned, ref_cnt = %s, in %s secs.'
                          % (total_cnt, ref_cnt, time.time() - timer_start))
        if raw_tw_info['tw_type'] != 't':
            continue
        tw_src_id = raw_tw_info['tw_src_id']
        if tw_src_id in df_merge.index and df_merge.loc[tw_src_id]['tw_raw_txt'] is not None:
            df_merge.at[tw_id, 'tw_raw_txt'] = None
            df_merge.at[tw_id, 'rt_ref'] = tw_src_id
            ref_cnt += 1
    logging.debug('[after_merge_processing] %s tws scanned, ref_cnt = %s, in %s secs.'
                  % (total_cnt, ref_cnt, time.time() - timer_start))
    return df_merge


def extract_raw_tw_info_wrapper(num_proc, job_id, ds_name, index_col):
    logging.debug('[extract_raw_tw_info_wrapper] Starts with %s procs on job %s.' % (int(num_proc), job_id))
    timer_start = time.time()

    multitasking(single_task_func=extract_raw_tw_info_single_proc,
                 single_task_params=None,
                 num_task=int(num_proc),
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_raw_txt_info_int_folder,
                 int_fmt=global_settings.g_raw_txt_info_int_fmt_regex,
                 after_merge_func=after_merge_processing,
                 out_path=global_settings.g_raw_txt_info_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)
    logging.debug('[extract_raw_tw_info_wrapper] All done in %s secs.' % str(time.time() - timer_start))


def gen_extract_raw_tw_info_tasks(num_proc, job_id):
    '''
    Task ids are of the format: job_id#x
    where x ranges in [0, num_proc-1]
    '''
    logging.debug('[gen_extract_raw_tw_info_tasks] Starts with %s procs on job %s.' % (num_proc, job_id))
    timer_start = time.time()

    done_cnt = 0
    l_tw_str = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_raw_data_folder):
        for filename in filenames:
            if re.match(r'.*\.txt', filename) is None:
                continue
            with open(dirpath + '/' + filename, 'r') as in_fd:
                for ln in in_fd:
                    ln = ln.strip()
                    if ln is None or len(ln) <= 1:
                        continue
                    l_tw_str.append(ln)
                in_fd.close()
            done_cnt += 1
            if done_cnt % 500 == 0 and done_cnt >= 500:
                logging.debug('[gen_extract_raw_tw_info_tasks] %s raw files scanned in %s secs.'
                              % (done_cnt, time.time() - timer_start))
    logging.debug('[gen_extract_raw_tw_info_tasks] %s raw files scanned in %s secs.'
                  % (done_cnt, time.time() - timer_start))
    num_tasks = len(l_tw_str)
    logging.debug('[gen_extract_raw_tw_info_tasks] %s tw objs.' % str(num_tasks))

    num_proc = int(num_proc)
    batch_size = math.ceil(num_tasks / num_proc)
    l_tasks = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_tasks.append(l_tw_str[i:i + batch_size])
        else:
            l_tasks.append(l_tw_str[i:])

    for idx, task in enumerate(l_tasks):
        task_id = str(job_id) + '#' + str(idx)
        with open(global_settings.g_raw_txt_info_task_file_fmt.format(task_id), 'w+') as out_fd:
            out_str = '\n'.join(task)
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[gen_extract_raw_tw_info_tasks] %s tasks are ready.' % len(l_tasks))


def group_raw_tw_info_by_country_code(ds_name):
    logging.debug('[group_raw_tw_info_by_country_code] Starts.')
    timer_start = time.time()

    df_raw_tw_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[group_raw_tw_info_by_country_code] Load in %s raw_tw_info recs.' % len(df_raw_tw_info))

    l_country_code = list(set(df_raw_tw_info['tw_place_country_code'].to_list()))
    logging.debug('[group_raw_tw_info_by_country_code] Found %s country codes: %s'
                  % (len(l_country_code), l_country_code))

    d_group = {country_code: [] for country_code in l_country_code}
    for tw_id, raw_tw_info_rec in df_raw_tw_info.iterrows():
        tw_country_code = raw_tw_info_rec['tw_place_country_code']
        d_group[tw_country_code].append((tw_id, raw_tw_info_rec['tw_type'], raw_tw_info_rec['tw_datetime'],
                                         raw_tw_info_rec['tw_lang'], raw_tw_info_rec['tw_usr_id'],
                                         raw_tw_info_rec['tw_src_id'], raw_tw_info_rec['tw_src_usr_id'],
                                         raw_tw_info_rec['tw_raw_txt'], raw_tw_info_rec['full_txt_flag'],
                                         raw_tw_info_rec['tw_place_id'], raw_tw_info_rec['tw_place_type'],
                                         raw_tw_info_rec['tw_place_full_name'],
                                         raw_tw_info_rec['tw_place_country_code']))

    for country_code in d_group:
        df_ready = pd.DataFrame(d_group[country_code], columns=['tw_id', 'tw_type', 'tw_datetime', 'tw_lang',
                                                                'tw_usr_id', 'tw_src_id', 'tw_src_usr_id',
                                                                'tw_raw_txt', 'full_txt_flag', 'tw_place_id',
                                                                'tw_place_type', 'tw_place_full_name',
                                                                'tw_place_country_code'])
        df_ready = df_ready.set_index('tw_id')
        if country_code == '' or country_code is None:
            country_code = 'UNKNOWN'
        pd.to_pickle(df_ready, global_settings.g_raw_txt_info_file_fmt.format(ds_name + '#' + country_code))
        logging.debug('[group_raw_tw_info_by_country_code] Done %s with %s tws.' % (country_code, len(df_ready)))
    logging.debug('[group_raw_tw_info_by_country_code] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        # gen_tasks 10 0
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        gen_extract_raw_tw_info_tasks(num_proc, job_id)
    elif cmd == 'raw_tw_info':
        # raw_tw_info 10 0 geo2021
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        ds_name = sys.argv[4]
        index_col = 'tw_id'
        extract_raw_tw_info_wrapper(num_proc, job_id, ds_name, index_col)
    elif cmd == 'group_by_country':
        # group_by_country geo2021
        ds_name = sys.argv[2]
        group_raw_tw_info_by_country_code(ds_name)
