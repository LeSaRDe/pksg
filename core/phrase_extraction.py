import logging
import traceback
import time
from os import path

import pandas as pd

from util import global_settings
from util.multitasking import multitasking


"""
PKSG PIPELINE
STAGE: PHRASE EXTRACTION (TXT_PH)

REQUIREMENTS:
    SEM_UNIT

OUTPUTS:
    text->phrase table: g_txt_phrase_file_fmt
        pandas DataFrame
        Index: text hash key (str): g_hash_txt_col
        Columns: phrases (list of tuples): g_txt_phrase_col
        
        Each phrase is a tuple:
        ([token_str_1, token_str_2], [POS_1, POS_2], [(token_1_start, token_1_end), (token_2_start, token_2_end)])
        or
        ([token_str], [POS], [(token_start, token_end)])
        where 'token_i_start' and 'token_i_end' are indexed relative to the (cleaned) text as a whole.
    
    ph_id->ph_str mapping: g_phrase_id_to_phrase_str_file_fmt
        pandas DataFrame
        Index: phrase id (int, starting with 0)
        Columns: phrase string (str)
        
        A phrase string consists of lemmas in the phrase, all lower-cased, without duplicate tokens, 
        and it is ordered by the lemmas.
        This mapping is one-to-one. 
        
    ph_str->ph_id mapping: g_phrase_str_to_phrase_id_file_fmt
        pandas DataFrame
        Index: phrase string (str)
        Columns: phrase id (int)
        
        This mapping is the reverse of the ph_id-ph_str mapping. 
    
    ph_id->text hash keys mapping: g_ph_id_to_txt_ids_file_fmt
        pandas DataFrame
        Index: phrase id (int)
        Columns: text hash key (list of ints)
        
        This mapping enumerates all texts that contain each of the phrases. 
    
    text hash key->ph_ids mapping: g_txt_id_to_ph_ids_file_fmt
        pandas DataFrame
        Index: text hash key (int)
        Columns: phrase id tuples (list of tuples)
        
        Each phrase id tuple consists of the phrase id and the pos tags:
        (ph_id, ph_pos_str)
        And ph_pos_str is a string consisting of ordered pos tags linked by whitespaces (i.e. ' ').
"""


def ph_to_ph_str(l_phrase_token):
    # l_phrase_token = phrase_tuple[0]
    phrase_str = ' '.join(sorted([token.lower() for token in l_phrase_token]))
    # l_phrase_token = functools.reduce(operator.iconcat, [sub_ph.split(' ') for sub_ph in l_phrase_token], [])
    # phrase_str = ' '.join(sorted(set([token.strip().lower() for token in l_phrase_token])))
    return phrase_str


def load_token_filter():
    '''
    A token can correspond to multiple POS tags. The token filter specifies particular POS tags accepted for a token.
    '''
    if not path.exists(global_settings.g_token_filter_file):
        return None
    d_token_filter = dict()
    with open(global_settings.g_token_filter_file, 'r') as in_fd:
        for ln in in_fd:
            fields = [ele.strip() for ele in ln.split(' ')]
            d_token_filter[fields[0]] = fields[1]
        in_fd.close()
    return d_token_filter


def extract_phrases_from_cls_graph(cls_graph, d_token_filter=None):
    if cls_graph is None:
        return None
    s_covered_nodes = []
    l_phrases = []
    try:
        for edge in cls_graph.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_txt = cls_graph.nodes(data=True)[node_1]['txt']
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_1_start = cls_graph.nodes(data=True)[node_1]['start']
            node_1_end = cls_graph.nodes(data=True)[node_1]['end']
            node_2_txt = cls_graph.nodes(data=True)[node_2]['txt']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            node_2_start = cls_graph.nodes(data=True)[node_2]['start']
            node_2_end = cls_graph.nodes(data=True)[node_2]['end']
            # phrase_start = min(node_1_start, node_2_start)
            # phrase_end = max(node_1_end, node_2_end)

            if d_token_filter is not None:
                l_node_1_token = [token.strip().lower() for token in node_1_txt.split(' ')]
                is_ok = True
                for token in l_node_1_token:
                    if token in d_token_filter and d_token_filter[token][0] != node_1_pos.lower()[0]:
                        is_ok = False
                        logging.debug('[extract_phrases_from_cls_json_str] skip %s.' % node_1_txt)
                        break
                if not is_ok:
                    continue
                l_node_2_token = [token.strip().lower() for token in node_2_txt.split(' ')]
                for token in l_node_2_token:
                    if token in d_token_filter and d_token_filter[token][0] != node_2_pos.lower()[0]:
                        is_ok = False
                        logging.debug('[extract_phrases_from_cls_json_str] skip %s.' % node_2_txt)
                        break
                if not is_ok:
                    continue

            phrase = ([node_1_txt, node_2_txt], [node_1_pos, node_2_pos],
                      [(node_1_start, node_1_end), (node_2_start, node_2_end)])
            s_covered_nodes.append(node_1)
            s_covered_nodes.append(node_2)
            l_phrases.append(phrase)
        s_covered_nodes = set(s_covered_nodes)
        if len(s_covered_nodes) < len(cls_graph.nodes):
            for node in cls_graph.nodes(data=True):
                if node[0] not in s_covered_nodes:
                    node_txt = node[1]['txt']
                    node_pos = node[1]['pos']
                    node_start = node[1]['start']
                    node_end = node[1]['end']
                    phrase = ([node_txt], [node_pos], [(node_start, node_end)])
                    l_phrases.append(phrase)
    except Exception as err:
        print('[extract_phrases_from_cls_json_str] %s' % err)
        traceback.print_exc()
    if len(l_phrases) > 0:
        return l_phrases
    return None


def extract_phrase_from_nps_str(nps, d_token_filter=None):
    if nps is None or len(nps) <= 0:
        return None
    nps = [([noun_phrase[0]], [noun_phrase[1]], [(noun_phrase[2], noun_phrase[3])]) for noun_phrase in nps]

    l_rm = []
    if d_token_filter is not None:
        for noun_phrase in nps:
            l_nps_token = [token.strip().lower() for token in noun_phrase[0][0].split(' ')]
            is_ok = True
            for token in l_nps_token:
                if token in d_token_filter and d_token_filter[token][0] != 'n':
                    is_ok = False
                    break
            if not is_ok:
                l_rm.append(noun_phrase)
        if len(l_rm) > 0:
            logging.debug('[extract_phrase_from_nps_str] rm %s phrases.' % str(len(l_rm)))
            for item in l_rm:
                nps.remove(item)
    return nps


def phrase_ext_single_task(task_id, hash_txt_col, phrase_col):
    logging.debug('[phrase_ext_single_task] Task %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_sem_unit = pd.read_pickle(global_settings.g_txt_phrase_task_file_fmt.format(task_id))
    logging.debug('[phrase_ext_single_task] Task %s: Load in %s sem unit recs.'
                  % (task_id, len(df_sem_unit)))

    d_token_filter = load_token_filter()

    l_ready = []
    for hash_txt, sem_unit_rec in df_sem_unit.iterrows():
        cls_graph = sem_unit_rec[global_settings.g_sem_unit_cls_col]
        nps = sem_unit_rec[global_settings.g_sem_unit_nps_col]
        l_ready_phrase = []
        l_cls_phrase = extract_phrases_from_cls_graph(cls_graph, d_token_filter)
        if l_cls_phrase is not None:
            l_ready_phrase += l_cls_phrase
        l_nps_phrase = extract_phrase_from_nps_str(nps, d_token_filter)
        if l_nps_phrase is not None:
            l_ready_phrase += l_nps_phrase
        if len(l_ready_phrase) > 0:
            l_ready.append((hash_txt, l_ready_phrase))
    df_phrase = pd.DataFrame(l_ready, columns=[hash_txt_col, phrase_col])
    pd.to_pickle(df_phrase, global_settings.g_txt_phrase_int_file_fmt.format(task_id))
    logging.debug('[phrase_ext_single_task] Task %s: All done with %s txt phrase recs in %s secs.'
                  % (task_id, len(df_phrase), time.time() - timer_start))


def map_between_txt_and_phrase(df_txt_phrase, phrase_col, ds_name):
    '''
    Each text contains a set of phrases. In df_txt_phrase, the phrases are listed by their actual tokens.
    We map the phrases into phrase_ids, and the order of phrases is preserved.
    '''
    logging.debug('[map_between_txt_and_phrase] Starts.')
    timer_start = time.time()

    # phrase_id_prefix = 'ph#'
    phrase_id_suffix = 0
    d_phrase_id_to_phrase_str = dict()
    d_phrase_str_to_phrase_id = dict()
    d_phrase_id_to_hash_txt = dict()
    d_hash_txt_to_phrase_id = dict()
    for hash_txt, phrase_rec in df_txt_phrase.iterrows():
        l_phrase_tuple = phrase_rec[phrase_col]
        for phrase_tuple in l_phrase_tuple:
            phrase_str = ph_to_ph_str(phrase_tuple[0])
            l_phrase_pos = phrase_tuple[1]
            phrase_pos = ' '.join(sorted(l_phrase_pos))
            if phrase_str not in d_phrase_str_to_phrase_id:
                # phrase_id = phrase_id_prefix + str(phrase_id_suffix)
                phrase_id = phrase_id_suffix
                phrase_id_suffix += 1
                d_phrase_str_to_phrase_id[phrase_str] = phrase_id
                d_phrase_id_to_phrase_str[phrase_id] = phrase_str
                d_phrase_id_to_hash_txt[phrase_id] = [hash_txt]
                if phrase_id_suffix % 10000 == 0 and phrase_id_suffix >= 10000:
                    logging.debug('[map_between_txt_and_phrase] Log in %s phrases and %s texts in %s secs.'
                                  % (len(d_phrase_id_to_hash_txt), len(d_hash_txt_to_phrase_id), time.time() - timer_start))
            else:
                phrase_id = d_phrase_str_to_phrase_id[phrase_str]
                if hash_txt not in d_phrase_id_to_hash_txt[phrase_id]:
                    d_phrase_id_to_hash_txt[phrase_id].append(hash_txt)
            # the order of phrases is preserved
            if hash_txt not in d_hash_txt_to_phrase_id:
                d_hash_txt_to_phrase_id[hash_txt] = [(phrase_id, phrase_pos)]
            else:
                d_hash_txt_to_phrase_id[hash_txt].append((phrase_id, phrase_pos))

    l_ready = []
    for phrase_id in d_phrase_id_to_phrase_str:
        l_ready.append((phrase_id, d_phrase_id_to_phrase_str[phrase_id]))
    df_phrase_id_to_phrase_str = pd.DataFrame(l_ready, columns=[global_settings.g_ph_id_col,
                                                                global_settings.g_ph_str_col])
    df_phrase_id_to_phrase_str = df_phrase_id_to_phrase_str.set_index(global_settings.g_ph_id_col)
    pd.to_pickle(df_phrase_id_to_phrase_str, global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[map_between_txt_and_phrase] df_phrase_id_to_phrase_str done with %s recs.'
                  % str(len(df_phrase_id_to_phrase_str)))

    l_ready = []
    for phrase_str in d_phrase_str_to_phrase_id:
        l_ready.append((phrase_str, d_phrase_str_to_phrase_id[phrase_str]))
    df_phrase_str_to_phrase_id = pd.DataFrame(l_ready, columns=[global_settings.g_ph_str_col,
                                                                global_settings.g_ph_id_col])
    df_phrase_str_to_phrase_id = df_phrase_str_to_phrase_id.set_index(global_settings.g_ph_str_col)
    pd.to_pickle(df_phrase_str_to_phrase_id, global_settings.g_phrase_str_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[map_between_txt_and_phrase] df_phrase_str_to_phrase_id done with %s recs.'
                  % str(len(df_phrase_str_to_phrase_id)))

    l_ready = []
    for phrase_id in d_phrase_id_to_hash_txt:
        l_ready.append((phrase_id, d_phrase_id_to_hash_txt[phrase_id]))
    df_ph_id_to_hash_txts = pd.DataFrame(l_ready, columns=[global_settings.g_ph_id_col, global_settings.g_hash_txt_col])
    df_ph_id_to_hash_txts = df_ph_id_to_hash_txts.set_index(global_settings.g_ph_id_col)
    pd.to_pickle(df_ph_id_to_hash_txts, global_settings.g_ph_id_to_hash_txts_file_fmt.format(ds_name))
    logging.debug('[map_between_txt_and_phrase] df_ph_id_to_hash_txts done with %s recs.'
                  % str(len(df_ph_id_to_hash_txts)))

    l_ready = []
    for hash_txt in d_hash_txt_to_phrase_id:
        l_ready.append((hash_txt, d_hash_txt_to_phrase_id[hash_txt]))
    df_hash_txt_to_ph_ids = pd.DataFrame(l_ready, columns=[global_settings.g_hash_txt_col, global_settings.g_ph_id_col])
    df_hash_txt_to_ph_ids = df_hash_txt_to_ph_ids.set_index(global_settings.g_hash_txt_col)
    pd.to_pickle(df_hash_txt_to_ph_ids, global_settings.g_hash_txt_to_ph_ids_file_fmt.format(ds_name))
    logging.debug('[map_between_txt_and_phrase] df_hash_txt_to_ph_ids done with %s recs.'
                  % str(len(df_hash_txt_to_ph_ids)))

    logging.debug('[map_between_txt_and_phrase] All done in %s secs.' % str(time.time() - timer_start))
    return df_txt_phrase


def phrase_ext_wrapper(ds_name, num_task, job_id, index_col=global_settings.g_hash_txt_col,
                       phrase_col=global_settings.g_txt_phrase_col):
    logging.debug('[phrase_ext_wrapper] Starts')
    timer_start = time.time()

    multitasking(single_task_func=phrase_ext_single_task,
                 single_task_params=(index_col, phrase_col),
                 num_task=num_task,
                 l_task_ids=None,
                 job_id=job_id,
                 task_type='proc',
                 int_folder=global_settings.g_txt_phrase_int_folder,
                 int_fmt=global_settings.g_txt_phrase_int_fmt_regex,
                 after_merge_func=map_between_txt_and_phrase,
                 after_merge_func_params=(phrase_col, ds_name),
                 out_path=global_settings.g_txt_phrase_file_fmt.format(ds_name),
                 index_col=index_col,
                 rm_int=False)

    logging.debug('[phrase_ext_wrapper] All done in %s secs.' % str(time.time() - timer_start))
