import logging
import re
import time
import json
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from gensim.parsing import preprocessing

import scenario_settings
import global_settings

from txt_cleansing import txt_clean_single_task
from txt_hash import txt_hash_single_task, txt_hash_after_merge
from semantic_units_extraction import sem_unit_ext_single_task
from semantic_units_extractor import SemUnitsExtractor, word_clean
from phrase_extraction import phrase_ext_single_task, map_between_txt_and_phrase
from sentiment_graphs import build_sgraph_from_json_single_task
from phrase_sentiment import phrase_sent_single_task
from pksg import pksg_single_task


def semeval2014task2_raw_txt_info(ds_idx, ds_name):
    ds_file_name = scenario_settings.g_raw_ds_names[ds_idx]
    logging.debug('[semeval2014task2_raw_txt_info] Starts with %s' % ds_file_name)

    xml_tree = ET.parse(global_settings.g_raw_data_folder + ds_file_name)
    xml_root = xml_tree.getroot()

    l_raw_txt_rec = []
    for sentence in xml_root:
        txt_id = int(sentence.get('id'))
        raw_txt = None
        l_term_pol = []
        for attr in sentence:
            if attr.tag == 'text':
                raw_txt = attr.text
            elif attr.tag == 'aspectTerms':
                for aspect in attr:
                    term = aspect.get('term')
                    pol = aspect.get('polarity')
                    l_term_pol.append((term, pol))
        l_raw_txt_rec.append((txt_id, raw_txt, l_term_pol))

    df_raw_txt = pd.DataFrame(l_raw_txt_rec, columns=[global_settings.g_txt_idx, global_settings.g_raw_txt_col,
                                                      'term_pol'])
    df_raw_txt = df_raw_txt.set_index(global_settings.g_txt_idx)
    df_raw_txt = df_raw_txt.loc[df_raw_txt.term_pol.apply(lambda x: len(x) > 0)]
    pd.to_pickle(df_raw_txt, global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_raw_txt_info] All done with %s raw_txt' % str(len(df_raw_txt)))


def remove_unncessary_dash(input_txt, l_dont_touch_terms):
    if '-' not in input_txt:
        return input_txt

    l_dont_touch_spans = []
    for dont_touch_term in l_dont_touch_terms:
        if '-' not in dont_touch_term:
            continue
        dont_touch_term = dont_touch_term.replace('(', '\(')
        dont_touch_term = dont_touch_term.replace(')', '\)')
        search_ret = re.search(dont_touch_term, input_txt)
        if search_ret is None:
            continue
        l_dont_touch_spans.append(search_ret.span())

    dash_ret = re.finditer('-', input_txt)
    l_rm_dash_pos = []
    for each_dash in dash_ret:
        each_dash_span = each_dash.span()
        each_dash_pos = each_dash_span[0]
        skip = False
        for dont_touch_span in l_dont_touch_spans:
            if dont_touch_span[0] <= each_dash_pos < dont_touch_span[1]:
                skip = True
        if not skip:
            l_rm_dash_pos.append(each_dash_pos)

    if len(l_rm_dash_pos) > 0:
        l_char = list(input_txt)
        for rm_dash_pos in l_rm_dash_pos:
            l_char[rm_dash_pos] = ' '
        clean_txt = ''.join(l_char)
        return clean_txt
    else:
        return input_txt


def semeval2014task2_txt_clean(ds_name):
    logging.debug('[semeval2014task2_txt_clean] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_raw_txt_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_txt_clean] Load in %s raw_txt_info recs.' % str(len(df_raw_txt_info)))

    df_raw_txt = df_raw_txt_info[[global_settings.g_raw_txt_col]]
    txt_clean_task_id = '0#0'
    pd.to_pickle(df_raw_txt, global_settings.g_txt_clean_task_file_fmt.format(txt_clean_task_id))

    txt_clean_single_task(txt_clean_task_id, global_settings.g_raw_txt_col,
                          clean_txt_col_name=global_settings.g_clean_txt_col)
    df_txt_clean = pd.read_pickle(global_settings.g_txt_clean_int_file_fmt.format(txt_clean_task_id))
    df_txt_clean = df_txt_clean.set_index(global_settings.g_txt_idx)
    l_txt_clean_rec = []
    for txt_id, txt_clean_rec in df_txt_clean.iterrows():
        txt_clean = txt_clean_rec[global_settings.g_clean_txt_col]
        l_term_pol = df_raw_txt_info.loc[txt_id]['term_pol']
        l_dont_touch_terms = [item[0] for item in l_term_pol]
        txt_clean = remove_unncessary_dash(txt_clean, l_dont_touch_terms)
        txt_clean = preprocessing.strip_multiple_whitespaces(txt_clean)
        txt_clean = txt_clean.strip()
        l_txt_clean_rec.append((txt_id, txt_clean))
    df_txt_clean = pd.DataFrame(l_txt_clean_rec, columns=[global_settings.g_txt_idx, global_settings.g_clean_txt_col])
    df_txt_clean = df_txt_clean.set_index(global_settings.g_txt_idx)
    pd.to_pickle(df_txt_clean, global_settings.g_txt_clean_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_txt_clean] All done with %s cleaned texts in %s secs.'
                  % (len(df_txt_clean), time.time() - timer_start))


def semeval2014task2_sem_unit(ds_name):
    logging.debug('[semeval2014task2_sem_unit] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_txt_clean = pd.read_pickle(global_settings.g_txt_clean_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_sem_unit] Load in %s cleaned texts.' % str(len(df_txt_clean)))

    df_raw_txt_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_sem_unit] Load in %s raw_txt_info recs.' % str(len(df_raw_txt_info)))

    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    sem_unit_task_id = '0#0'

    l_sem_unit = []
    for txt_id, txt_clean_rec in df_txt_clean.iterrows():
        if txt_id == 3508:
            print('bk')
        l_texts = []
        clean_txt = txt_clean_rec[global_settings.g_clean_txt_col]
        l_texts.append((txt_id, clean_txt))
        l_term_pol = df_raw_txt_info.loc[txt_id]['term_pol']
        l_cust_ph = [item[0] for item in l_term_pol]
        df_sem_units = sem_unit_ext_ins.sem_unit_extraction_batch(l_texts, sem_unit_task_id, global_settings.g_txt_idx,
                                                                  l_cust_ph)
        l_sem_unit.append(df_sem_units)
    df_sem_unit = pd.concat(l_sem_unit)
    df_sem_unit = df_sem_unit.set_index(global_settings.g_txt_idx)
    pd.to_pickle(df_sem_unit, global_settings.g_sem_unit_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_sem_unit] All done with %s sem units in %s secs.' %
                  (len(df_sem_unit), time.time() - timer_start))


def semeval2014task2_stanford_sentiments_preparation(ds_name):
    logging.debug('[semeval2014task2_stanford_sentiments_preparation] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_txt_clean = pd.read_pickle(global_settings.g_txt_clean_file_fmt.format(ds_name))
    df_txt_clean = df_txt_clean.loc[df_txt_clean[global_settings.g_clean_txt_col].notnull()]
    logging.debug('[semeval2014task2_stanford_sentiments_preparation] Load in %s cleaned texts.'
                  % str(len(df_txt_clean)))

    l_json_out = []
    for hash_txt_key, clean_txt_rec in df_txt_clean.iterrows():
        clean_txt = clean_txt_rec[global_settings.g_clean_txt_col]
        l_json_out.append({global_settings.g_txt_idx: str(hash_txt_key), global_settings.g_clean_txt_col: clean_txt})

    txt_sent_job_id = 0
    txt_sent_task_id = 0
    with open(global_settings.g_txt_sent_task_file_fmt.format(str(txt_sent_job_id) + '#' + str(txt_sent_task_id)),
              'w+') as out_fd:
        json.dump(l_json_out, out_fd)
        out_fd.close()
    logging.debug('[semeval2014task2_stanford_sentiments_preparation] All done in %s secs.'
                  % str(time.time() - timer_start))


def semeval2014task2_txt_ph(ds_name):
    logging.debug('[semeval2014task2_txt_ph] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_sem_unit = pd.read_pickle(global_settings.g_sem_unit_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_txt_ph] Load in %s sem units.' % str(len(df_sem_unit)))

    txt_ph_task_id = '0#0'
    pd.to_pickle(df_sem_unit, global_settings.g_txt_phrase_task_file_fmt.format(txt_ph_task_id))
    phrase_ext_single_task(txt_ph_task_id, global_settings.g_txt_idx, global_settings.g_txt_phrase_col)
    df_txt_ph = pd.read_pickle(global_settings.g_txt_phrase_int_file_fmt.format(txt_ph_task_id))
    df_txt_ph = df_txt_ph.set_index(global_settings.g_txt_idx)
    pd.to_pickle(df_txt_ph, global_settings.g_txt_phrase_file_fmt.format(ds_name))
    map_between_txt_and_phrase(df_txt_ph, global_settings.g_txt_phrase_col, ds_name)

    logging.debug('[semeval2014task2_txt_ph] All done in %s secs.' % str(time.time() - timer_start))


def semeval2014task2_sgraph(ds_name):
    logging.debug('[semeval2014task2_sgraph] Starts with %s.' % ds_name)
    timer_start = time.time()

    sgraph_task_id = '0#0'
    build_sgraph_from_json_single_task(sgraph_task_id, global_settings.g_txt_idx, global_settings.g_sgraph_col)
    df_sgraph = pd.read_pickle(global_settings.g_sgraph_int_file_fmt.format(sgraph_task_id))
    df_sgraph = df_sgraph.set_index(global_settings.g_txt_idx)
    pd.to_pickle(df_sgraph, global_settings.g_sgraph_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_sgraph] All done in %s secs.' % str(time.time() - timer_start))


def semeval2014task2_ph_sent(ds_name):
    logging.debug('[semeval2014task2_ph_sent] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_sgraph = pd.read_pickle(global_settings.g_sgraph_file_fmt.format(ds_name))
    num_sgraph = len(df_sgraph)
    logging.debug('[semeval2014task2_ph_sent] Load in %s sgraph recs.' % str(num_sgraph))

    df_phrase = pd.read_pickle(global_settings.g_txt_phrase_file_fmt.format(ds_name))
    num_phrase = len(df_phrase)
    logging.debug('[semeval2014task2_ph_sent] Load in %s phrase recs.' % str(num_phrase))

    l_task_ready = []
    for hash_txt, phrase_rec in df_phrase.iterrows():
        if hash_txt not in df_sgraph.index:
            continue
        phrase = phrase_rec[global_settings.g_txt_phrase_col]
        sgarph = df_sgraph.loc[hash_txt][global_settings.g_sgraph_col]
        l_task_ready.append((hash_txt, phrase, sgarph))

    ph_sent_task_id = '0#0'
    df_task = pd.DataFrame(l_task_ready, columns=[global_settings.g_hash_txt_col, global_settings.g_txt_phrase_col,
                                                  global_settings.g_sgraph_col])
    df_task = df_task.set_index(global_settings.g_hash_txt_col)
    pd.to_pickle(df_task, global_settings.g_phrase_sent_task_file_fmt.format(ph_sent_task_id))

    phrase_sent_single_task(ph_sent_task_id, global_settings.g_hash_txt_col, global_settings.g_txt_phrase_col,
                            global_settings.g_sgraph_col, global_settings.g_ph_sent_col,
                            global_settings.g_ph_sent_leftover_col)

    df_ph_sent = pd.read_pickle(global_settings.g_phrase_sent_int_file_fmt.format(ph_sent_task_id))
    df_ph_sent = df_ph_sent.set_index(global_settings.g_hash_txt_col)
    pd.to_pickle(df_ph_sent, global_settings.g_phrase_sent_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_ph_sent] All done in %s secs.' % str(time.time() - timer_start))


def semeval2014task2_pksg(ds_name):
    logging.debug('[semeval2014task2_pksg] Starts with %s.' % ds_name)
    timer_start = time.time()

    df_phrase = pd.read_pickle(global_settings.g_txt_phrase_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_pksg] load in df_phrase with %s recs in %s secs.' %
                  (len(df_phrase), time.time() - timer_start))

    df_ph_sent = pd.read_pickle(global_settings.g_phrase_sent_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_pksg] load in df_ph_sent with %s recs in %s secs.'
                  % (len(df_ph_sent), time.time() - timer_start))

    df_sgraph = pd.read_pickle(global_settings.g_sgraph_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_pksg] load in df_sgraph with %s recs in %s secs.'
                  % (len(df_sgraph), time.time() - timer_start))

    df_hash_txt_to_ph_ids = pd.read_pickle(global_settings.g_hash_txt_to_ph_ids_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_pksg] load in df_hash_txt_to_ph_ids with %s recs in %s secs.'
                  % (len(df_hash_txt_to_ph_ids), time.time() - timer_start))

    l_hash_txt = list(set(df_phrase.index.to_list()).intersection(df_ph_sent.index.to_list())
                      .intersection(df_sgraph.index.to_list()))

    pksg_task_id = '0#0'
    with open(global_settings.g_pksg_task_file_fmt.format(pksg_task_id), 'w+') as out_fd:
        out_str = '\n'.join([str(item) for item in l_hash_txt])
        out_fd.write(out_str)
        out_fd.close()

    pksg_single_task(pksg_task_id, sim_ph_id=None, df_phrase=df_phrase, df_txt_id_to_ph_ids=df_hash_txt_to_ph_ids,
                     df_ph_sent=df_ph_sent, df_sgraph=df_sgraph)

    df_pksg = pd.read_pickle(global_settings.g_pksg_int_file_fmt.format(pksg_task_id))
    df_pksg = df_pksg.set_index(global_settings.g_hash_txt_col)
    pd.to_pickle(df_pksg, global_settings.g_pksg_file_fmt.format(ds_name))
    logging.debug('[semeval2014task2_pksg] All done in %s secs.' % str(time.time() - timer_start))


def aggregate_node_sentiments(node_label, pksg, df_ph_id_to_ph_str):
    if node_label not in pksg:
        raise Exception('[aggregate_node_sentiments] Node %s is not in PKSG.' % str(node_label))

    d_sent_cat_cnt = {i: [0, []] for i in range(5)}
    for neig in pksg.neighbors(node_label):
        neig_str = df_ph_id_to_ph_str.loc[neig][global_settings.g_ph_str_col]
        edge = pksg.edges[(node_label, neig)]
        l_sent = edge['sent']
        for sent_vec in l_sent:
            sent_cat = np.argmax(np.asarray(sent_vec))
            d_sent_cat_cnt[sent_cat][0] += 1
            d_sent_cat_cnt[sent_cat][1].append(neig_str)

    d_ret = {'negative': d_sent_cat_cnt[0][1] + d_sent_cat_cnt[1][1],
             'positive': d_sent_cat_cnt[3][1] + d_sent_cat_cnt[4][1],
             'conflict': d_sent_cat_cnt[0][1] + d_sent_cat_cnt[1][1] + d_sent_cat_cnt[3][1] + d_sent_cat_cnt[4][1],
             'neutral': d_sent_cat_cnt[2][1]}
    l_ret = []
    if np.sum([d_sent_cat_cnt[i][0] for i in range(5)]) > 0:
        dom_sent_cnt = np.max([d_sent_cat_cnt[i][0] for i in range(5)])
        l_dom_sent_cat = [i for i, cnt in enumerate([d_sent_cat_cnt[i][0] for i in range(5)]) if cnt == dom_sent_cnt]
        # dom_sent_cat = np.argmax([d_sent_cat_cnt[i][0] for i in range(5)])
        for dom_sent_cat in l_dom_sent_cat:
            if dom_sent_cat == 0 or dom_sent_cat == 1:
                l_ret.append('negative')
            elif dom_sent_cat == 3 or dom_sent_cat == 4:
                l_ret.append('positive')
            else:
                l_ret.append('neutral')
        if 'positive' in l_ret and 'negative' in l_ret:
            l_ret = ['conflict']
        return l_ret, d_ret
    else:
        return ['neutral'], d_ret


def aggregate_ph_sentiments(term, l_ph_sent):
    d_sent_cat_cnt = {i: [0, []] for i in range(5)}
    for ph_sent in l_ph_sent:
        if term in ph_sent[0] and len(ph_sent[0]) > 1:
            sent_vec = ph_sent[3]
            sent_cat = np.argmax(np.asarray(sent_vec))
            d_sent_cat_cnt[sent_cat][0] += 1
            d_sent_cat_cnt[sent_cat][1].append(ph_sent[0])

    d_ret = {'negative': d_sent_cat_cnt[0][1] + d_sent_cat_cnt[1][1],
             'positive': d_sent_cat_cnt[3][1] + d_sent_cat_cnt[4][1],
             'conflict': d_sent_cat_cnt[0][1] + d_sent_cat_cnt[1][1] + d_sent_cat_cnt[3][1] + d_sent_cat_cnt[4][1],
             'neutral': d_sent_cat_cnt[2][1]}
    l_ret = []
    if np.sum([d_sent_cat_cnt[i][0] for i in range(5)]) > 0:
        dom_sent_cnt = np.max([d_sent_cat_cnt[i][0] for i in range(5)])
        l_dom_sent_cat = [i for i, cnt in enumerate([d_sent_cat_cnt[i][0] for i in range(5)]) if cnt == dom_sent_cnt]
        # dom_sent_cat = np.argmax([d_sent_cat_cnt[i][0] for i in range(5)])
        for dom_sent_cat in l_dom_sent_cat:
            if dom_sent_cat == 0 or dom_sent_cat == 1:
                l_ret.append('negative')
            elif dom_sent_cat == 3 or dom_sent_cat == 4:
                l_ret.append('positive')
            else:
                l_ret.append('neutral')
        if 'positive' in l_ret and 'negative' in l_ret:
            l_ret = ['conflict']
        return l_ret, d_ret
    else:
        return ['neutral'], d_ret


def semeval2014task2_term_sent(ds_name):
    logging.debug('[semeval2014task2_term_sent] Starts.')

    df_pksg = pd.read_pickle(global_settings.g_pksg_file_fmt.format(ds_name))
    df_raw_txt_info = pd.read_pickle(global_settings.g_raw_txt_info_file_fmt.format(ds_name))
    df_ph_id_to_ph_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    df_ph_str_to_ph_id = pd.read_pickle(global_settings.g_phrase_str_to_phrase_id_file_fmt.format(ds_name))
    df_ph_sent = pd.read_pickle(global_settings.g_phrase_sent_file_fmt.format(ds_name))
    # sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)

    l_ret_rec = []
    for txt_id, raw_txt_info_rec in df_raw_txt_info.iterrows():
        pksg = df_pksg.loc[txt_id]['pksg']
        l_ph_sent = df_ph_sent.loc[txt_id]['ph_sent']
        raw_txt = raw_txt_info_rec['raw_txt']
        term_pol = raw_txt_info_rec['term_pol']
        for term, pol in term_pol:
            # term_str = ' '.join(sorted([token.lower() for token in term.split(' ')]))
            term_str = term.lower()
            term_ph_id = df_ph_str_to_ph_id.loc[term_str][global_settings.g_ph_id_col]
            # l_ret, d_ret = aggregate_node_sentiments(term_ph_id, pksg, df_ph_id_to_ph_str)
            l_ret, d_ret = aggregate_ph_sentiments(term, l_ph_sent)
            confusion = False
            if len(l_ret) > 1:
                confusion = True
            sent_match = False
            for ret in l_ret:
                sent_str = ret
                l_neig_str = d_ret[sent_str]
                sent_match = False
                if pol == sent_str:
                    sent_match = True
                    break
            potential_match = None
            if not sent_match and len(d_ret[pol]) > 0:
                potential_match = True
            ret_rec_id = str(txt_id) + '#' + term
            l_ret_rec.append((ret_rec_id, sent_match, confusion, potential_match, sent_str, l_neig_str, pol, raw_txt, d_ret))
    df_rec = pd.DataFrame(l_ret_rec, columns=['term_id', 'sent_match', 'confusion', 'potential', 'sent', 'evidence',
                                              'truth', 'raw_txt', 'd_ret'])
    df_rec = df_rec.set_index('term_id')
    pd.to_pickle(df_rec, global_settings.g_work_folder + 'ret.pickle')
    logging.debug('[semeval2014task2_term_sent] All done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ds_idx = 0
    ds_name = 'restaurants'

    # semeval2014task2_raw_txt_info(ds_idx, ds_name)
    # semeval2014task2_txt_clean(ds_name)
    # semeval2014task2_sem_unit(ds_name)
    # semeval2014task2_stanford_sentiments_preparation(ds_name)
    # semeval2014task2_txt_ph(ds_name)
    # semeval2014task2_sgraph(ds_name)
    # semeval2014task2_ph_sent(ds_name)
    # semeval2014task2_pksg(ds_name)
    semeval2014task2_term_sent(ds_name)
