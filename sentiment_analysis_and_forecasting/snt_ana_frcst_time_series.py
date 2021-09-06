import logging
import re
import time
import sys
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scenario_settings
import global_settings


"""
PKSG PIPELINE
STAGE: TIME SERIES (T_INT)

REQUIREMENTS:
    RAW_TXT, PKSG

OUTPUTS:
    A time interval table: g_t_int_file_fmt
        pandas DataFrame
        Index: time interval id (int): g_t_int_id
        Columns: 
            time interval start (str): g_t_int_start_col, format: YYYYMMDDHHMMSS
            time interval end (str): g_t_int_end_col, format: YYYYMMDDHHMMSS
            text ids within the time interval: (list of strs): g_t_int_txt_ids
    
    A PKSG table for each time interval: g_pksg_ts_file_fmt
        (same as g_pksg_file_fmt)
    
    A txt file storing all time interval PKSGs' names: g_pksg_ts_ds_name_file_fmt
"""


def gen_time_series(ds_name, t_int_len, t_stride, t_start=None, t_end=None):
    """
    Generate a time series over a given data. The time points are sampled from the data. Each time point corresponds
    to a time interval centered at the time point or ahead of the time point.
    :param t_int_len: (timedelta) The length of each time interval, e.g. 5 days.
    :param t_stride: (timedelta) The length of gap between two consecutive time points.
    :param t_start: (str) The start time of the time series. The format is: g_datetime_fmt
    :param t_end: (str) The end time of the time series. The format is: g_datetime_fmt
    :return: (pandas DataFrame) A time series stored in a table.
    Index: (int) time interval index, starting at 0.
    Columns:
        t_int_start (str): the start of the time interval
        t_int_end (str): the end of the time interval
        txt_id (list): the list of text ids within the time interval
    """
    timer_start = time.time()
    df_raw_tw_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[gen_time_series] Load in %s raw txt recs from %s.' % (len(df_raw_tw_info), ds_name))

    # Determine time series boundaries
    t_min = min(df_raw_tw_info['tw_datetime'].to_list())
    t_max = max(df_raw_tw_info['tw_datetime'].to_list())
    if t_start is None or t_start < t_min:
        t_start = t_min
    if t_end is None or t_end > t_max:
        t_end = t_max
    t_start = datetime.strptime(t_start, global_settings.g_datetime_fmt)
    t_end = datetime.strptime(t_end, global_settings.g_datetime_fmt)

    if t_start >= t_end:
        raise Exception('[gen_time_series] t_start %s is ahead of t_end %s !' % (t_start, t_end))
    if (t_end - t_start) < t_stride:
        raise Exception('[gen_time_series] t_stride %s is greater than (t_end - t_start) %s !'
                        % (t_stride, (t_end - t_start)))

    # Obtain time intervals
    l_t_int = []
    cur_t_int_start = t_start
    overflowed = False
    while True:
        cur_t_int_end = cur_t_int_start + t_int_len
        if cur_t_int_end <= t_end:
            l_t_int.append((cur_t_int_start.strftime(global_settings.g_datetime_fmt),
                            cur_t_int_end.strftime(global_settings.g_datetime_fmt)))
        elif cur_t_int_start < t_end and not overflowed:
            l_t_int.append((cur_t_int_start.strftime(global_settings.g_datetime_fmt),
                            t_end.strftime(global_settings.g_datetime_fmt)))
            overflowed = True
        else:
            break
        cur_t_int_start = cur_t_int_start + t_stride

    # Collect txt_ids for each time interval
    d_t_int_rec = {idx: [] for idx in range(len(l_t_int))}
    df_raw_tw_info = df_raw_tw_info.sort_values('tw_datetime')
    cur_t_int_idx = 0
    for txt_id, txt_rec in df_raw_tw_info.iterrows():
        txt_datetime = txt_rec['tw_datetime']
        for t_int_idx in range(cur_t_int_idx, len(l_t_int)):
            cur_t_int_start = l_t_int[t_int_idx][0]
            cur_t_int_end = l_t_int[t_int_idx][1]
            if cur_t_int_start <= txt_datetime <= cur_t_int_end:
                d_t_int_rec[t_int_idx].append(txt_id)
            if cur_t_int_end <= txt_datetime:
                cur_t_int_idx = t_int_idx
            if cur_t_int_start > txt_datetime:
                break

    # Output
    l_t_int_rec = []
    for t_int_idx in range(len(l_t_int)):
        l_t_int_rec.append((t_int_idx, l_t_int[t_int_idx][0], l_t_int[t_int_idx][1], d_t_int_rec[t_int_idx]))
    df_t_int = pd.DataFrame(l_t_int_rec, columns=[global_settings.g_t_int_id, global_settings.g_t_int_start_col,
                                                  global_settings.g_t_int_end_col, global_settings.g_t_int_txt_ids])
    df_t_int = df_t_int.set_index(global_settings.g_t_int_id)
    pd.to_pickle(df_t_int, global_settings.g_t_int_file_fmt.format(ds_name))
    logging.debug('[gen_time_series] All done with %s time intervals in %s secs'
                  % (len(df_t_int), time.time() - timer_start))


def convert_timedelta_str_to_timedelta(timedelta_str):
    """
    :param timedelta_str: (str) The timedelta string is of the format:
    weeks(any length digits)#days(0 or 1 digit)#hours(0-2 digits)#minutes(0-2 digits)#seconds(0-2 digits)
    r'\d*#\d?#\d{0,2}#\d{0,2}#\d{0,2}'
    """
    if re.match(r'\d*#\d?#\d{0,2}#\d{0,2}#\d{0,2}', timedelta_str) is None:
        raise Exception('[convert_timedelta_str_to_timedelta] Invalid timedelta_str: %s' % timedelta_str)

    l_fields = timedelta_str.split('#')
    weeks = int(l_fields[0]) if l_fields[0] != '' else 0
    days = int(l_fields[1]) if l_fields[1] != '' else 0
    hours = int(l_fields[2]) if l_fields[2] != '' else 0
    minutes = int(l_fields[3]) if l_fields[3] != '' else 0
    seconds = int(l_fields[4]) if l_fields[4] != '' else 0
    return timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)


def pksg_for_time_series(t_int_ds_name, pksg_ds_name):
    timer_start = time.time()
    df_t_int = pd.read_pickle(global_settings.g_t_int_file_fmt.format(t_int_ds_name))
    logging.debug('[pksg_for_time_series] Load in %s time intervals.' % str(len(df_t_int)))

    df_pksg = pd.read_pickle(global_settings.g_pksg_file_fmt.format(pksg_ds_name))
    logging.debug('[pksg_for_time_series] Load in %s pksg.' % str(len(df_pksg)))
    s_pksg_hash_txts = set(df_pksg.index.to_list())

    df_txt_id_to_hash_txt = pd.read_pickle(global_settings.g_txt_id_to_hash_txt_file_fmt.format(t_int_ds_name))
    logging.debug('[pksg_for_time_series] Load in %s txt_id_to_hash_txt recs from %s.'
                  % (len(df_txt_id_to_hash_txt), t_int_ds_name))

    l_pksg_ts_ds_name = []
    for t_int_id, t_int_rec in df_t_int.iterrows():
        t_int_start = t_int_rec[global_settings.g_t_int_start_col]
        t_int_end = t_int_rec[global_settings.g_t_int_end_col]
        t_int_txt_ids = t_int_rec[global_settings.g_t_int_txt_ids]
        pksg_ts_ds_name = t_int_start + '#' + t_int_end + '@' + pksg_ds_name
        if len(t_int_txt_ids) <= 0:
            logging.error('[pksg_for_time_series] No text between %s and %s!' % (t_int_start, t_int_end))
            continue
        else:
            t_int_hash_txts = df_txt_id_to_hash_txt.loc[df_txt_id_to_hash_txt.index.intersection(t_int_txt_ids)][global_settings.g_hash_txt_col].to_list()
            t_int_hash_txts = set(t_int_hash_txts).intersection(s_pksg_hash_txts)
            if len(t_int_hash_txts) <= 0:
                logging.error('[pksg_for_time_series] No PKSG rec between %s and %s!' % (t_int_start, t_int_end))
                continue
            df_pksg_ts = df_pksg.loc[t_int_hash_txts]
            pd.to_pickle(df_pksg_ts, global_settings.g_pksg_ts_file_fmt.format(pksg_ts_ds_name))
            l_pksg_ts_ds_name.append(pksg_ts_ds_name)
            logging.debug('[pksg_for_time_series] Output pksg ts %s with %s pksg.'
                          % (pksg_ts_ds_name, len(df_pksg_ts)))
    if len(l_pksg_ts_ds_name) <= 0:
        logging.error('[pksg_for_time_series] No pksg ts!')
    else:
        with open(global_settings.g_pksg_ts_ds_name_file_fmt.format(pksg_ds_name), 'w+') as out_fd:
            out_str = '\n'.join(l_pksg_ts_ds_name)
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[pksg_for_time_series] All done with %s pksg ts in %s secs.'
                  % (len(l_pksg_ts_ds_name), time.time() - timer_start))


def classify_binary_sentiment(sent_vec):
    if sent_vec is None:
        return None

    sent_class = np.argmax(np.asarray(sent_vec))
    if sent_class == 0 or sent_class == 1:
        return 0
    elif sent_class == 2:
        return None
    elif sent_class == 3 or sent_class == 4:
        return 1


def pksg_ts_stats(ds_name):
    logging.debug('[pksg_ts_stats] Starts.')
    plt.set_loglevel('error')

    df_merge_pksg_ts = pd.read_pickle(global_settings.g_merge_pksg_ts_file_fmt.format(ds_name))
    logging.debug('[pksg_ts_stats] Load in %s merge_pksg_ts.' % str(len(df_merge_pksg_ts)))

    df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[pksg_ts_stats] Load in %s recs for df_ph_id_to_ph_str' % str(len(df_ph_id_to_ph_str)))

    l_t_int = [t_int.split('#')[0] for t_int in df_merge_pksg_ts.index.to_list()]
    l_node_ts = [pksg.nodes(data=True) for pksg in df_merge_pksg_ts['merge_pksg'].to_list()]
    l_edge_ts = [pksg.edges(data=True) for pksg in df_merge_pksg_ts['merge_pksg'].to_list()]

    # Nodes & Edges Stats
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    axes.grid(True)
    axes.set_title('202101-202106 PKSG TS Nodes & Edges Stats', fontsize=15, fontweight='semibold')
    l_num_nodes = [len(nodes) for nodes in l_node_ts]
    l_num_edges = [len(edges) for edges in l_edge_ts]
    axes.plot(l_num_nodes, linewidth=2, label='Nodes')
    axes.plot(l_num_edges, linewidth=2, label='Edges')
    axes.set_xticks([i for i in range(len(l_t_int))])
    axes.set_xticklabels(l_t_int, rotation=45, ha='right', rotation_mode="anchor")
    axes.set_yticks([i for i in range(0, max(l_num_edges) + 20000, 20000)])
    axes.legend()
    plt.tight_layout(pad=1.0)
    plt.show()
    plt.clf()
    plt.close()

    # Top Frequent Phrases
    l_node_freq_ts = []
    for l_nodes in l_node_ts:
        l_node_freq = []
        for node in l_nodes:
            node_freq = np.sum(list(node[1]['pos'].values()))
            node_str = df_ph_id_to_ph_str.loc[node[0]]['ph_str']
            l_node_freq.append((node_str, node_freq))
        l_node_freq = sorted(l_node_freq, key=lambda k: k[1], reverse=True)
        l_node_freq_ts.append(l_node_freq)

    for t_int_idx, t_int in enumerate(l_t_int):
        print(t_int + ':' + ','.join([item[0] + '@' + str(item[1]) for item in l_node_freq_ts[t_int_idx][:50]]))

    # Top Degree Phrases
    l_node_deg_ts = []
    for t_int_name, pksg_ts_rec in df_merge_pksg_ts.iterrows():
        t_int = t_int_name.split('#')[0]
        merge_pksg = pksg_ts_rec[global_settings.g_merge_pksg_col]
        l_node_deg = sorted([(df_ph_id_to_ph_str.loc[node][global_settings.g_ph_str_col], merge_pksg.degree[node]) for node in merge_pksg.nodes()],
                            key=lambda k: k[1], reverse=True)
        l_node_deg_ts.append(l_node_deg)

    for t_int_idx, t_int in enumerate(l_t_int):
        print(t_int + ':' + ','.join([item[0] + '@' + str(item[1]) for item in l_node_deg_ts[t_int_idx][:50]]))

    # Sentiment Coordinates for Top Degree Phrases
    l_node_sent_ts = []
    for t_int_name, pksg_ts_rec in df_merge_pksg_ts.iterrows():
        t_int = t_int_name.split('#')[0]
        merge_pksg = pksg_ts_rec[global_settings.g_merge_pksg_col]
        l_node_deg = sorted([node for node in merge_pksg.nodes()], key=lambda k: merge_pksg.degree[k], reverse=True)
        l_node_deg = l_node_deg[:100]
        l_node_sent = []
        for node in l_node_deg:
            np_sent_cnt_vec = np.zeros(2)
            l_neig = merge_pksg.neighbors(node)
            total_weight = 0
            for neig in l_neig:
                edge_weight = merge_pksg.edges[(node, neig)]['weight']
                total_weight += edge_weight
                l_sent = merge_pksg.edges[(node, neig)]['sent']
                for sent_vec in l_sent:
                    sent_cat = classify_binary_sentiment(sent_vec)
                    if sent_cat is not None:
                        np_sent_cnt_vec[sent_cat] += 1
            l_node_sent.append((df_ph_id_to_ph_str.loc[node][global_settings.g_ph_str_col], np_sent_cnt_vec, total_weight))
        l_node_sent_ts.append((t_int, l_node_sent))

    for t_int, l_node_sent in l_node_sent_ts:
        year = t_int[:4]
        month = t_int[4:6]
        day = t_int[6:8]
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
        axes.grid(True)
        axes.set_title('%s-%s-%s PKSG TS Top 50 Node Sentiments' % (year, month, day),
                       fontsize=15, fontweight='semibold')

        l_x_coord = []
        l_y_coord = []
        l_str = []
        l_color = []
        l_mark_size = []
        for node_str, node_sent_vec, total_weight in l_node_sent:
            # if node_str == 'covid vaccine':
            #     continue
            if np.sum(node_sent_vec) > 0:
                color = node_sent_vec / np.sum(node_sent_vec)
                color = color[1]
            else:
                color = 0
            if total_weight > 0:
                node_sent_vec = node_sent_vec / total_weight
                node_sent_vec += 0.000000000001
                # node_sent_vec = - np.log(node_sent_vec / total_weight)
                node_sent_vec = node_sent_vec * 100
            l_x_coord.append(node_sent_vec[0])
            l_y_coord.append(node_sent_vec[1])
            l_str.append(node_str)
            l_color.append(color)
            l_mark_size.append(5000 * np.sum(node_sent_vec) / total_weight)

        axes.scatter(l_x_coord, l_y_coord, s=l_mark_size, c=l_color, cmap='cool')
        for idx, node_str in enumerate(l_str):
            axes.annotate(node_str, (l_x_coord[idx], l_y_coord[idx]), fontsize=15)
        axes.set_xlabel('Negative', fontsize=15, fontweight='semibold')
        axes.set_ylabel('Postive', fontsize=15, fontweight='semibold')
        plt.tight_layout(pad=1.0)
        plt.savefig('/home/mf3jh/workspace/data/pksg_ts_stats/%s-%s-%s_phrase_sentiments.PNG' % (year, month, day),
                    format='PNG')
        plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_time_series':
        # gen_time_series 202005 ###5# ###1#
        ds_name = sys.argv[2]
        t_int_len = sys.argv[3]
        t_int_len = convert_timedelta_str_to_timedelta(t_int_len)
        if t_int_len == timedelta(0):
            raise Exception('[gen_time_series] t_int_len is 0!')
        t_stride = sys.argv[4]
        t_stride = convert_timedelta_str_to_timedelta(t_stride)
        if t_stride == timedelta(0):
            raise Exception('[gen_time_series] t_stride is 0!')
        if len(sys.argv) >= 6:
            t_start = sys.argv[5]
        if len(sys.argv) >= 7:
            t_end = sys.argv[6]
        gen_time_series(ds_name, t_int_len, t_stride, t_start=None, t_end=None)
    elif cmd == 'pksg_ts':
        # pksg_ts 0 ca
        job_id = sys.argv[2]
        ds_name = sys.argv[3]
        t_int_ds_name = ds_name
        pksg_ds_name = ds_name + '#' + job_id
        pksg_for_time_series(t_int_ds_name, pksg_ds_name)
    elif cmd == 'pksg_ts_stats':
        # pksg_ts_stats va2021
        ds_name = sys.argv[2]
        pksg_ts_stats(ds_name)
