'''
OBJECTIVES:
    Define the class of semantic units extractor.
'''
import logging
import json
import re
from os import path, walk
from collections import deque
import threading
import math
import time
import sqlite3

import spacy
from gensim.parsing import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import global_settings
import sd_2_usd
from stopwords import STOP_WORDS


def prelim_txt_clean(raw_text, extra_clean_func=None, extra_clean_func_params=None):
    '''
    This function primarily removes urls, hasded ids, tags and trivial texts. It does NOT fully clean the input text.
    By utilizing SemUnitsExtractor, the actual text cleansing is done at the token level, and has been embedded in the
    semantic unit extraction.
    '''
    if raw_text is None or raw_text == '':
        return None
    l_clean_sents = []
    l_dirty_sents = raw_text.split('\n')
    for raw_dirt_sent in l_dirty_sents:
        # remove url
        clean_text = re.sub(r'url: [\S]*', '', raw_dirt_sent)
        clean_text = re.sub(r'http[\S]*', '', clean_text)
        # remove hashed ids
        clean_text = re.sub(r'RT\s', ' ', clean_text)
        clean_text = re.sub(r'@un:\s[\S]{22}\s', ' ', clean_text)
        clean_text = re.sub(r'@[^\s]+', ' ', clean_text)
        clean_text = re.sub(r'\s[\S]{22}\s', ' ', clean_text)
        # extra clean
        if extra_clean_func is not None:
            if extra_clean_func_params is not None:
                extra_clean_func_params = (clean_text,) + extra_clean_func_params
                clean_text = extra_clean_func(*extra_clean_func_params)
            else:
                clean_text = extra_clean_func(clean_text)
        # remove # symbol
        clean_text = re.sub(r'#', ' ', clean_text)
        # remove unnecessary white spaces
        clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
        clean_text = clean_text.strip()
        trivial_test = re.match(r'.*[a-zA-A]*', clean_text)
        if trivial_test is not None:
            l_clean_sents.append(clean_text)
    return '\n'.join(l_clean_sents)


def word_clean(word):
    '''
    Clean the input token.
    1. Remove all non-word characters
    2. Remove HTML and other tags
    3. Remove punctuations
    4. Remove unnecessary white spaces
    5. Remove numbers
    6. Remove short tokens
    :param
        word: A token.
    :return:
        A cleaned token.
    '''
    clean_text = re.sub(r'[^\w\s]', ' ', word)
    clean_text = preprocessing.strip_tags(clean_text)
    clean_text = preprocessing.strip_punctuation(clean_text)
    clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
    clean_text = preprocessing.strip_numeric(clean_text)
    if len(clean_text) <= 3:
        return ''
    # clean_text = preprocessing.strip_short(clean_text, minsize=2)
    return clean_text


def word_remove_punctuation(word):
    clean_text = preprocessing.strip_punctuation(word)
    clean_text = clean_text.strip()
    return clean_text


class SemUnitsExtractor:
    def __init__(self, config_file_path):
        if config_file_path is None or config_file_path == '':
            raise Exception('[SemUnitsExtractor:__init__] Cannot find configuration file!')
        with open(config_file_path, 'r') as in_fd:
            d_conf = json.load(in_fd)
            in_fd.close()
        try:
            self.m_conf_spacy_model_name = d_conf['spacy_model']
            self.m_use_custom_stopwords = d_conf['use_custom_stopwords']
            self.m_conf_stopwords_path = d_conf['stopwords_path']
            self.m_conf_spacy_coref_greedyness = float(d_conf['spacy_coref_greedyness'])
            self.m_s_ner_tags = set(d_conf['ner'])
            self.m_b_en_dep_filter = bool(d_conf['en_dep_filter'])
            self.m_s_core_deps = set(d_conf['core_dep'])
            self.m_b_keep_neg = bool(d_conf['keep_neg'])
            self.m_s_neg_cues = set(d_conf['neg_cue'])
            self.m_spacy_model_ins = self.spacy_init(self.m_conf_spacy_model_name, self.m_conf_spacy_coref_greedyness)
            self.m_s_stopwords = self.load_stopwords(self.m_conf_stopwords_path)
            self.m_d_sd_2_usd = sd_2_usd.g_d_sd_to_usd
        except Exception as err:
            logging.error('[SemUnitsExtractor:__init__] ' + err)
            return

    def spacy_init(self, spacy_model_name, coref_greedyness):
        '''
        Load the spaCy model with a specific name.
        :param coref_greedyness:
        :return:
        '''
        spacy_model = spacy.load(spacy_model_name)
        # neuralcoref.add_to_pipe(spacy_model, greedyness=coref_greedyness)
        return spacy_model

    def load_stopwords(self, stopwords_path):
        if self.m_use_custom_stopwords:
            return None
        if not path.exists(stopwords_path):
            raise Exception('[SemUnitsExtractor:__init__] No stopwords file!')
        s_stopwords = set([])
        with open(stopwords_path, 'r') as in_fd:
            ln = in_fd.readline()
            while ln:
                sw = ln.strip()
                s_stopwords.add(sw)
                ln = in_fd.readline()
            in_fd.close()
        return s_stopwords

    def is_trivial_dep(self, token_dep):
        if self.m_b_en_dep_filter:
            if token_dep in self.m_s_core_deps:
                return False
            else:
                return True
        else:
            return False

    def is_stopword(self, token):
        if not self.m_use_custom_stopwords:
            if token in self.m_spacy_model_ins.Defaults.stop_words:
                return True
            else:
                return False
        else:
            if token in self.m_s_stopwords:
                return True
            else:
                return False

    def is_trivial_token(self, spacy_token, cleaned_lemma=None):
        '''
        Determine if the input token is trivial.
        NOTE:
        This method is not perfect. Any token in a sentence with a capped leading letter can be recognized as an
        entity. In this case, this method cannot detect if the token is trivial.
        NOTE:
        Before calling this function, the dep of the input token should be changed to neg if it is a negation token.
        TODO:
        Any better method to address the above issue?
        How should we deal with special entities? A capped token can be easily a false special entity.
        :param
            spacy_token: A spaCy annotated token.
            cleaned_lemma: Provides the cleaned lemma of 'spacy_token'.
            If 'cleaned_lemma' is None, 'spacy_token' is cleaned here.
        :return:
            True -- trivial
            False -- non-trivial
        '''
        if cleaned_lemma is None:
            clean_token = word_clean(spacy_token.lemma_)
        else:
            clean_token = cleaned_lemma
        if clean_token == '':  # or not clean_token.isascii():
            return True
        if self.m_b_keep_neg:
            if self.is_stopword(clean_token.lower()) and spacy_token.dep_ != 'neg':
                return True
        else:
            if self.is_stopword(clean_token.lower()):
                return True
        return False

    def spacy_pipeline_parse(self, raw_text):
        '''
        Obtain POS, NER, Dep Parse Trees, Noun Chunks and other linguistic features from the text. The text is firstly
        segmented into sentences, and then parsed. spaCy uses Universal Dependencies and POS tags.
        :param
            text: a raw text that may contains multiple sentences.
        :return:
            A list of tagged sentences, and the parsed doc.
        TODO:
            SpaCy 2.3 seems not so right working with Python 3.8. Also, the 'pipe' method in 'Language' returns a
            Doc generator rather than a straight Doc, which makes it awkward to extract sentences. If we don't use
            'pipe', we lose our chance to take advantage of 'n_process' (v2.3) or 'n_threads' (v2.2). As an alternative,
            we multiprocess our sem unit extraction procedure, and each proc runs with a single proc SpaCy instance.
            We may need to test the compatibility of SpaCy furthermore and find better solution of parallelism.
        '''
        if raw_text is None:
            return None, None
        # l_sents = []
        doc = self.m_spacy_model_ins(raw_text, disable=['ner'])
        # parse_doc_gen = self.m_spacy_model_ins.pipe(raw_text, disable=['ner'], n_process=10)
        # for doc in parse_doc_gen:
        #     l_sents += list(doc.sents)
        # doc = self.m_spacy_model_ins.pipe(raw_text, disable=['ner'], n_process=10)
        return list(doc.sents), doc

    def extract_sal_lemmas_from_sent(self, raw_txt):
        '''
        Extract non-trivial lemmas from raw_txt.
        '''
        l_spacy_sents, spacy_doc = self.spacy_pipeline_parse(raw_txt)
        l_sal_lemmas = []
        for sent_id, spacy_sent in enumerate(l_spacy_sents):
            for word in spacy_sent:
                word_lemma = word_clean(word.lemma_)
                if not self.is_trivial_token(word) \
                        and word_lemma not in l_sal_lemmas:
                    l_sal_lemmas.append(word_lemma)
        return l_sal_lemmas

    # @profile
    def spacy_extract_phrases_from_sent(self, sent_id, spacy_sent, d_cust_ph=None):
        '''
        1. Detect all custom phrases. Each token is at most covered by one custom phrase.
           Exact matches (i.e. original tokens, case sensitive, token order) are performed in searching for
           custom phrases. A custom phrase can be a singleton token.
        2. Detect all noun phrases. A noun phrase should not overlap with any custom phrase. If an overlap happens,
           then the noun phrase is dropped. When searching for noun phrases, tokens are converted to lemmas, duplicate
           tokens are dropped, trivial tokens are dropped, and the kept tokens are ordered.
           A noun phrase needs to contain at least two tokens.
        3. Indicate each token's belonging to the detected phrases if any.
        :param
            spacy_sent: A spaCy parsed sentence. Should not be None.
            d_cust_ph: (dict, key=str, val=list of tuples) Custom phrases that need to be preserved.
            Case insensitive, token order matters, no lemmatisation.
            No noun phrase should overlap with any custom phrase.
            {'ph token#1': [(ph#1, num of tokens), (ph#2, num of tokens), ...]}
        :return:
            1. A dict of phrase info.
               {phrase_id@int: (phrase_string@str, phrase_POS@str, phrase_start@int, phrase_end@int), ...}
               -- 'phrase_id' is only meaningful within 'spacy_sent' starting with 0.
               -- 'phrase_string' is linked phrase tokens (or lemmas) by blank spaces.
               For custom phrases, 'phrase_string' is exactly the string preserved from the input without lemmatisation.
               For noun phrases, phrase tokens are converted to lemmas, and duplicate lemmas are removed. Additionally,
               the kept lemmas are ordered alphabetically.
               -- 'phrase_POS' marks 'NOUN' for noun phrases and 'CUST' for custom phrases.
               -- 'phrase_start' is the starting character position of the phrase.
               -- 'phrase_end' is the ending character position of the phrase (i.e. the offset right after the phrase).
            2. A dict of token belonging info.
               {token_i@int : phrase_id:int}
               -- 'token_i' is the index of a token. Only tokens covered by a phrase is keyed.
               -- 'phrase_id' is the same as above.
            Note that the token indices stored in the outputs all should be absolute indices.
        '''
        d_phrase = dict()
        d_token_belong = dict()

        if spacy_sent is None or len(spacy_sent) <= 0:
            return d_phrase, d_token_belong

        ph_id = 0
        # Note that 'cur_token_i' is the relative token index in 'spacy_sent'. The actual token index is
        # 'cur_token_i + space_sent.start'.
        cur_token_i = 0
        if d_cust_ph is not None:
            while cur_token_i < len(spacy_sent):
                # Find custom phrases in spacy_sent.
                # Custom phrases are disjoint. Tokens in custom phrases will NOT be substituted by noun phrases if any.
                token_txt = spacy_sent[cur_token_i].text.lower()
                # token_txt = word_remove_punctuation(token_txt)
                if token_txt in d_cust_ph:
                    for cust_ph, num_cust_ph_tokens in d_cust_ph[token_txt]:
                        cur_ph_end_i = cur_token_i + num_cust_ph_tokens
                        if cur_ph_end_i <= len(spacy_sent) and \
                                cust_ph == spacy_sent[cur_token_i: cur_ph_end_i].text.lower():
                            d_phrase[ph_id] = (cust_ph, 'CUST',
                                               spacy_sent[cur_token_i].idx,
                                               spacy_sent[cur_ph_end_i - 1].idx + len(spacy_sent[cur_ph_end_i - 1]))
                            while cur_token_i < cur_ph_end_i:
                                if cur_token_i + spacy_sent.start not in d_token_belong:
                                    d_token_belong[cur_token_i + spacy_sent.start] = ph_id
                                else:
                                    logging.error(
                                        '[spacy_extract_phrases_from_sent] A custom phrase overlap occurs at sentence:%s, token: %s, full sentence:%s'
                                        % (sent_id, spacy_sent[cur_token_i].text, spacy_sent))
                                cur_token_i += 1
                            ph_id += 1
                            cur_token_i = cur_ph_end_i - 1
                            break
                cur_token_i += 1

        for noun_phrase in spacy_sent.noun_chunks:
            if len([token.i for token in noun_phrase if token.i in d_token_belong]) > 0:
                continue
            l_token_i = [token.i for token in noun_phrase]
            if len(l_token_i) <= 1:
                continue
            l_sal_lemmas = []
            for token in noun_phrase:
                token_lemma = word_clean(token.lemma_)
                if not self.is_trivial_token(token, cleaned_lemma=token_lemma):
                    l_sal_lemmas.append(token_lemma.lower())
            s_sal_lemmas = set(l_sal_lemmas)
            # If a phrase contains less than 2 tokens, it's not a phrase. And it must have contained in the core
            # clause graph.
            if len(s_sal_lemmas) <= 1:
                continue
            d_phrase[ph_id] = (' '.join(sorted(s_sal_lemmas)), 'NP',
                               spacy_sent[min(l_token_i) - spacy_sent.start].idx,
                               spacy_sent[max(l_token_i) - spacy_sent.start].idx)
            for token_i in l_token_i:
                if token_i not in d_token_belong:
                    d_token_belong[token_i] = ph_id
                else:
                    logging.error('[spacy_extract_phrases_from_sent] A noun phrase overlap occurs at sentence:%s, token: %s, full sentence:%s'
                                  % (sent_id, spacy_sent[token_i - spacy_sent.start].text, spacy_sent))
            ph_id += 1
        return d_phrase, d_token_belong

    # def nps_to_np_info(self, l_nps):
    #     """
    #     Return format:
    #     {token index: (noun phrase string, set of rest token indices in the noun phrase), ...}
    #     Only used by 'extract_cls_from_sent'.
    #     """
    #     d_np_info = dict()
    #     for np_tup in l_nps:
    #         np_str = np_tup[0]
    #         s_token_idx = np_tup[1]
    #         for token_i in s_token_idx:
    #             if token_i not in d_np_info:
    #                 d_np_info[token_i] = (np_str, s_token_idx.remove(token_i))
    #             else:
    #                 logging.error('[nps_to_np_info] Token: %s appears in multiple noun phrases: %s, %s.'
    #                               % (token_i, d_np_info[token_i][0], np_str))
    #     return d_np_info

    def spacy_extract_npool_from_sent(self, spacy_sent):
        '''
        Extract non-trivial nouns from a spacy parsed sentence. Stopwords are removed. All resulting nouns are converted to
        lemmas.
        :param
            spacy_sent: A spacy parsed sentence. Should not be None.
        :return:
            A set of non-trivial nouns.
        '''
        s_npool = set([])
        for token in spacy_sent:
            if token.pos_ == 'NOUN' and not token.is_stop and token.lemma_ not in self.m_s_stopwords:
                s_npool.add(token.lemma_)
        return s_npool

    # def token_to_np(self, spacy_token, l_nps, cleaned_lemma=None):
    #     '''
    #     Substitude a token with a noun phrase if any. All inputs should be the same sentence.
    #     :param
    #         spacy_token: a spaCy token.
    #         l_nps: a list of noun phrase tuples.
    #     :return:
    #         A tuple of the format:
    #         (cleaned phrase string, POS, set of indices of phrase tokens, start char position of phrase,
    #         end char position of phrase)
    #         If the input token is contained in a noun phrase, then return the noun phrase string with 'NOUN' as its
    #         POS tag. Otherwise, return the cleaned token with its original POS tag.
    #     '''
    #     if spacy_token is None:
    #         raise Exception('[SemUnitsExtractor:token_to_np] spacy_token is None!')
    #
    #     if cleaned_lemma is None:
    #         token_str = self.word_clean(spacy_token.lemma_)
    #     else:
    #         token_str = cleaned_lemma
    #     if len(l_nps) <= 0:
    #         return token_str, spacy_token.pos_, {spacy_token.i}, spacy_token.idx, \
    #                spacy_token.idx + len(spacy_token)
    #
    #     ph_str = token_str
    #     token_i = spacy_token.i
    #     for np_tpl in l_nps:
    #         if token_i in np_tpl[1]:
    #             ph_str = np_tpl[0]
    #             return ph_str, 'NOUN', np_tpl[1], np_tpl[2], np_tpl[3]
    #     return ph_str, spacy_token.pos_, {spacy_token.i}, spacy_token.idx, spacy_token.idx + len(spacy_token)

    def neg_tag_substitute(self, spacy_sent):
        '''
        Substitute the dependency tags of negation tokens to 'neg'. Due to the imperfection of dependency parsers,
        some negation tokens, e.g. 'neither', may not be tagged as 'neg'. We fix this. The negation token list comes
        from the paper 'Negation Scope Detection for Twitter Sentiment Analysis'. Also, we add 'noone', 'couldnt',
        'wont' and 'arent' that are not included in this paper but in "Sentiment Symposium Tutorial" by Potts 2011.
        :param
            spacy_sent: A spaCy parsed sentence.
        :return:
            A modified spaCy sentence with all considered negation tokens tagged as 'neg'.
        '''
        for token in spacy_sent:
            if token.text in self.m_s_neg_cues:
                token.dep_ = 'neg'
        return spacy_sent

    def build_node_id(self, sent_id, node_start, node_end):
        return sent_id + '|' + str(node_start) + '|' + str(node_end)

    def find_nearest_ancestor_in_nx_graph(self, spacy_token, d_node_info, d_nearest_ancestor, nx_dep_tree):
        """
        Return the nearest ancestor of 'spacy_token' in 'nx_dep_tree' if any. Also, update 'd_nearest_ancestor' if any.
        'spacy_token' must NOT be trivial.
        'd_nearest_ancestor' is a compression data structure for the ancestor searching.
            Key: token index in the sentence
            Value: Ancestor node label
        As some tokens may not be added into 'nx_dep_tree' (e.g. trivial tokens), some paths in the dependency parse
        tree need to be compressed.
        Note that we make 'conj' tokens share the same ancestor.
        """
        if spacy_token.i not in d_node_info:
            raise Exception('[find_nearest_ancestor_in_nx_graph] Token %s is not in d_node_info!' % str(spacy_token))

        conj_jump = False
        if spacy_token.dep_ == 'conj':
            conj_jump = True

        cur_token = spacy_token
        parent = cur_token.head
        while parent != cur_token:
            if cur_token.i in d_nearest_ancestor:
                d_nearest_ancestor[spacy_token.i] = d_nearest_ancestor[cur_token.i]
                return d_nearest_ancestor[cur_token.i]

            if parent.i not in d_node_info:
                cur_token = parent
                parent = cur_token.head
                continue

            parent_label = d_node_info[parent.i][0]
            if parent_label in nx_dep_tree.nodes:
                if conj_jump or d_node_info[spacy_token.i][0] == parent_label:
                    cur_token = parent
                    parent = cur_token.head
                    conj_jump = False
                    continue

                d_nearest_ancestor[spacy_token.i] = parent_label
                return parent_label
            else:
                cur_token = parent
                parent = cur_token.head
        d_nearest_ancestor[spacy_token.i] = None
        return None

    # @profile
    def extract_cls_from_sent(self, sent_id, spacy_sent, d_phrase, d_token_belong):
        '''
        Extract core clause structures from a spacy parsed sentence. Each structure is represented as a graph (a tree
        without root). Each graph contains salient clause structures led by verbs. These structures are inspired by the
        five clause structures of simple sentences:
          1. <intransitive_verb subj>
          2. <transitive_verb subj obj>
          3. <transitive_verb subj iobj dobj>
          4. <subj subj_comp>
          5. <transitive_verb subj obj obj_comp>
        The dependency relations involved in these clause structures are various, and are (partially) enumerated in
        'g_s_core_deps'. For tokens in a noun phrase, substitute the noun phrase for the tokens. We always add lemmas
        as vertices into the resulting graph.
        NOTE:
        We temporarily use the dependency parser in spaCy to do this. Though, it may not be the best solution. Also,
        the parsing results from the spaCy parser may not the same as the visualization from their online demo.
        TODO:
        We may need to compare between the spaCy parser and the biaffine attention parser in AllenNLP to get the better
        one, or we may design a "ensemble" parser taking advantage of both of them.
        TODO:
        1. Negation: We would like to attach a negation word to its most dependent word. This dependent word can be a
        verb, noun, adjective or something else.
        # 2. conj
        3. More precise non-trivial words
        4. Co-references so far seem rather unstable in performance. Also, since a message may contain some content that
        cannot be easily resolved (e.g. pictures and videos), it may lead to further misleading to co-reference
        resolution. Thus, we temporarily do not do this.
        :param
            spacy_sent: A spaCy parsed sentence.
            d_phrase: Output from 'spacy_extract_phrases_from_sent'.
            d_token_belong: Output from 'spacy_extract_phrases_from_sent'.
        :return:
            A NetworkX undirected and unweighted graph that represents the core clause structures. Vertices are strings.
            Edges are induced (i.e. not exactly) by dependencies.
            The node ids in the resulting graph are unique ids rather than actual texts on the nodes. The node labels
            are the actual texts.
        NOTE:
        A vertex in the resulting graph is a composed string of the format: [tw_id]|[sent_idx]|[token]
        '''
        if spacy_sent is None:
            raise Exception('[SemUnitsExtractor:extract_cls_from_sent] spacy_sent is invalid!')

        nx_cls = nx.Graph()
        # Find ROOT
        root = spacy_sent.root
        # Prepare node info, and add nodes.
        # d_node_info data structure:
        # (node label string,
        #  node lemma string,
        #  node POS string,
        #  node start character position,
        #  node end character position)
        # Only non-trivial tokens are listed in d_node_info.
        d_node_info = dict()
        for token in spacy_sent:
            node_str = None
            if token.i in d_token_belong:
                node_str = d_phrase[d_token_belong[token.i]][0]
                node_pos = d_phrase[d_token_belong[token.i]][1]
                node_start = d_phrase[d_token_belong[token.i]][2]
                node_end = d_phrase[d_token_belong[token.i]][3]
            else:
                token_lemma = word_clean(token.lemma_)
                if not self.is_trivial_dep(sd_2_usd.sd_to_usd(token.dep_)) \
                        and not self.is_trivial_token(token, cleaned_lemma=token_lemma):
                    node_str = token_lemma
                    node_pos = token.pos_
                    node_start = token.idx
                    node_end = token.idx + len(token)
            if node_str is not None:
                node_str = node_str.lower()
                node_label = self.build_node_id(sent_id, node_start, node_end)
                d_node_info[token.i] = (node_label, node_str, node_pos, node_start, node_end)
                if node_label not in nx_cls.nodes:
                    nx_cls.add_node(node_label, txt=node_str, pos=node_pos, type='root', start=node_start, end=node_end)

        # Find all equivalent conj nodes, and link conj nodes.
        d_label_to_conj = dict()
        d_conj_group = dict()
        conj_group_id = 0
        for token_i in d_node_info:
            token = spacy_sent[token_i - spacy_sent.start]
            if token.dep_ != 'conj':
                continue
            parent = token.head
            if parent.i not in d_node_info:
                continue
            token_node_label = d_node_info[token_i][0]
            conj_node_label = d_node_info[parent.i][0]
            # conj can happen within a phrase
            if token_node_label == conj_node_label:
                continue
            if token_node_label not in d_label_to_conj and conj_node_label not in d_label_to_conj:
                d_label_to_conj[token_node_label] = conj_group_id
                d_label_to_conj[conj_node_label] = conj_group_id
                d_conj_group[conj_group_id] = [token_node_label, conj_node_label]
                conj_group_id += 1
            else:
                if token_node_label in d_label_to_conj:
                    d_label_to_conj[conj_node_label] = d_label_to_conj[token_node_label]
                    d_conj_group[d_label_to_conj[token_node_label]].append(conj_node_label)
                elif conj_node_label in d_label_to_conj:
                    d_label_to_conj[token_node_label] = d_label_to_conj[conj_node_label]
                    d_conj_group[d_label_to_conj[conj_node_label]].append(token_node_label)
            nx_cls.add_edge(token_node_label, conj_node_label)

        # Traverse the dep tree, and add edges.
        q_nodes = deque()
        for child in root.children:
            q_nodes.append(child)

        d_nearest_ancestor = dict()
        while len(q_nodes) > 0:
            cur = q_nodes.pop()
            if cur.i in d_node_info:
                cur_node_label = d_node_info[cur.i][0]
                # If the current node is non-trivial and it's conjunctive to another non-trivial node (i.e. typically
                # its immediate parent), then the children of the current node should also be linked to its conjunction.
                nearest_ancestor = self.find_nearest_ancestor_in_nx_graph(cur, d_node_info, d_nearest_ancestor, nx_cls)
                if nearest_ancestor is not None and nearest_ancestor != cur_node_label:
                    nx_cls.nodes(data=True)[cur_node_label]['type'] = 'node'
                    nx_cls.add_edge(cur_node_label, nearest_ancestor)
                    if nearest_ancestor in d_label_to_conj:
                        for conj_node_label in d_conj_group[d_label_to_conj[nearest_ancestor]]:
                            nx_cls.add_edge(cur_node_label, conj_node_label)
            for child in cur.children:
                q_nodes.append(child)

        l_roots = [node[0] for node in nx_cls.nodes(data=True) if node[1]['type'] == 'root']
        for i in range(0, len(l_roots) - 1):
            for j in range(i + 1, len(l_roots)):
                nx_cls.add_edge(l_roots[i], l_roots[j])

        if len(nx_cls.nodes) == 1 and list(nx_cls.nodes)[0] in self.m_s_neg_cues:
            return None

        return nx_cls

    # @profile
    def extract_sem_units_from_text(self, raw_txt, txt_id, l_cust_ph=None):
        '''
        Extract semantic units for a given piece of raw text.
        :param
            raw_txt: A piece of raw text which can contain multiple sentences.
            txt_id: The unique ID for the raw_txt.
            l_cust_ph: (list of str) The list of custom phrases.
        :return:
            The core clause structure graph (the union graph from all sentences), the list of noun phrases.
        TODO:
        We may add other semantic units.
        '''
        if raw_txt is None or len(raw_txt) == 0:
            logging.debug('[SemUnitsExtractor:extract_sem_units_from_text] A trivial text occurs.')
            return None, None

        d_cust_ph = None
        if l_cust_ph is not None:
            d_cust_ph = dict()
            for phrase in l_cust_ph:
                # l_token = [token.strip().lower() for token in phrase.split(' ')]
                l_token = [token.text.lower() for token in self.m_spacy_model_ins(phrase)]
                if l_token[0] not in d_cust_ph:
                    # d_cust_ph[l_token[0]] = [l_token[1:]]
                    d_cust_ph[l_token[0]] = [(phrase.lower(), len(l_token))]
                else:
                    # d_cust_ph[l_token[0]].append(l_token[1:])
                    d_cust_ph[l_token[0]].append((phrase.lower(), len(l_token)))

            for lead_token in d_cust_ph:
                d_cust_ph[lead_token] = sorted(d_cust_ph[lead_token], key=lambda k: k[1], reverse=True)

        l_nps = []
        nx_cls = nx.Graph()
        l_spacy_sents, spacy_doc = self.spacy_pipeline_parse(raw_txt)
        for sent_id, spacy_sent in enumerate(l_spacy_sents):
            d_phrase, d_token_belong = self.spacy_extract_phrases_from_sent(str(txt_id) + '|' + str(sent_id),
                                                                            spacy_sent, d_cust_ph)
            l_nps += [d_phrase[key] for key in d_phrase]
            nx_cls = self.union_nx_graphs(nx_cls,
                                          self.extract_cls_from_sent(str(txt_id) + '|' + str(sent_id),
                                                                     spacy_sent, d_phrase, d_token_belong))
        return nx_cls, l_nps

    def union_nx_graphs(self, nx_1, nx_2):
        if nx_2 is None:
            logging.error('[SemUnitsExtractor:union_nx_graphs] nx_2 is None!')
            if nx_1 is None:
                raise Exception('[SemUnitsExtractor:union_nx_graphs] nx_1 or nx_2 is empty!')
            else:
                return nx_1
        for node in nx_2.nodes(data=True):
            nx_1.add_node(node[0], txt=node[1]['txt'], pos=node[1]['pos'], start=node[1]['start'], end=node[1]['end'])
        for edge in nx_2.edges:
            nx_1.add_edge(edge[0], edge[1])
        return nx_1

    def output_nx_graph(self, nx_graph, draw_img=False, save_img=False, fig_path=None):
        if nx_graph is None or len(nx_graph.nodes) == 0:
            return
        plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 200, 'h_pad': 200, 'rect': None})
        pos = nx.spring_layout(nx_graph, k=0.8)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        d_node_labels = {node[0]: node[1]['txt'] + ':' + node[1]['pos'] for node in nx_graph.nodes(data=True)}
        nx.draw_networkx_nodes(nx_graph, pos, node_size=50)
        nx.draw_networkx_labels(nx_graph, pos, labels=d_node_labels, font_size=25, font_family="sans-serif")
        nx.draw_networkx_edges(nx_graph, pos, width=2, edge_color='b')
        if save_img and fig_path is not None:
            plt.savefig(fig_path, format="PNG")
        if draw_img:
            plt.show()
        plt.clf()
        plt.close()

    def task_multithreads(self, op_func, l_tasks, num_threads, job_id, output_format=None, output_db_path=None,
                          en_draw=False, other_params=()):
        '''
        A multithreading wrapper for a list a task with texts.
        :param
            op_func: The thread function to process a subset of tasks.
            l_tasks: A list of tasks. Each task is of the format: (task_id, text)
            num_threads: The number of threads
            output_folder: For outputs
            en_draw: True - draw outputs if necessary
        :return:
            No direct return value but outputs in the output folder and draws if any.
        '''
        timer_1 = time.time()
        logging.debug('[SemUnitsExtractor:task_multithreads] %s tasks in total.' % len(l_tasks))
        batch_size = math.ceil(len(l_tasks) / num_threads)
        l_l_subtasks = []
        for i in range(0, len(l_tasks), batch_size):
            if i + batch_size < len(l_tasks):
                l_l_subtasks.append(l_tasks[i:i + batch_size])
            else:
                l_l_subtasks.append(l_tasks[i:])
        logging.debug('[SemUnitsExtractor:task_multithreads] %s threads.' % len(l_l_subtasks))

        l_threads = []
        t_id = 0
        for l_each_batch in l_l_subtasks:
            t = threading.Thread(target=op_func, args=(l_each_batch, job_id + '_' + str(t_id), output_format,
                                                       en_draw) + other_params)
            t.setName('t_mul_task_' + str(t_id))
            t.start()
            l_threads.append(t)
            t_id += 1

        while len(l_threads) > 0:
            for t in l_threads:
                if t.is_alive():
                    t.join(1)
                else:
                    l_threads.remove(t)
                    logging.debug('[SemUnitsExtractor:task_multithreads] Thread %s is finished.' % t.getName())

        if output_db_path is not None:
            self.output_sem_units_to_db(output_format, output_db_path)
        logging.debug('[SemUnitsExtractor:task_multithreads] All done in %s sec for %s tasks.'
                      % (time.time() - timer_1, len(l_tasks)))

    def sem_unit_extraction_batch(self, l_tasks, task_id, index_name, l_cust_ph=None, en_indexed_output=False,
                                  output_format=None, en_draw=False):
        '''
        Extract semantic units for a list of tasks with texts.
        TODO:
        Add modifier structures and noun pools.
        :param
            l_tasks: A list of tasks with texts, each of which is of the format: (txt_id, text)
            task_id: Task name
            index_name: The name of the index. The output table will be indexed on this name.
            l_cust_ph: Custom phrases.
            en_indexed_output: True -- index the output by 'index_name'.
            output_folder: For outputs
            en_draw: Draw outputs if necessary
        :return:
            Return a table file of the pickle format including cls_graph (NetworkX Graph) and nps (list). The table
            is indexed by the txt_id sent in by l_tasks.
        '''
        # logging.debug('[SemUnitsExtractor:sem_unit_extraction_batch] Task %s: Starts with %s tasks.'
        #               % (task_id, len(l_tasks)))
        timer_start = time.time()
        cnt = 0
        l_out_recs = []
        for txt_id, sent_txt in l_tasks:
            nx_cls, l_nps = self.extract_sem_units_from_text(sent_txt, str(txt_id) + '|' + str(cnt), l_cust_ph)
            if (nx_cls is None or len(nx_cls.nodes()) <= 0) and (l_nps is None or len(l_nps) <= 0):
                continue
            if len(nx_cls.nodes()) <= 0:
                cls_graph = None
            else:
                cls_graph = nx_cls
                # cls_str = json.dumps(nx.adjacency_data(nx_cls))
            if en_draw:
                self.output_nx_graph(nx_cls, output_format + txt_id + '_cls_graph.png')
            if len(l_nps) <= 0:
                nps = None
            # else:
            #     nps = '\n'.join([item[0] + '|' + str(item[2]) + '|' + str(item[3]) for item in l_nps])
            out_rec = (txt_id, cls_graph, l_nps)
            l_out_recs.append(out_rec)
            cnt += 1
            if cnt % 5000 == 0 and cnt >= 5000:
                logging.debug('[SemUnitsExtractor:sem_unit_extraction_batch] Task %s: %s texts are done in %s secs.'
                              % (task_id, cnt, str(time.time() - timer_start)))
        logging.debug('[SemUnitsExtractor:sem_unit_extraction_batch] Task %s: All %s texts are done in %s secs.'
                      % (task_id, cnt, str(time.time() - timer_start)))
        out_df = pd.DataFrame(l_out_recs, columns=[index_name, global_settings.g_sem_unit_cls_col,
                                                   global_settings.g_sem_unit_nps_col])
        if en_indexed_output:
            out_df = out_df.set_index(index_name)
        if output_format is not None:
            out_df.to_pickle(output_format.format(str(task_id)))
        # logging.debug('[SemUnitsExtractor:sem_unit_extraction_batch] %s: All done with %s recs in %s secs.'
        #               % (task_id, len(out_df), str(time.time() - timer_start)))
        return out_df

    def output_sem_units_to_db(self, sem_unit_folder, db_path):
        db_conn = sqlite3.connect(db_path)
        db_cur = db_conn.cursor()
        sql_str = '''create table if not exists ven_tw_sem_units (tw_id text primay key, cls_json_str text, nps_str text)'''
        db_cur.execute(sql_str)
        sql_str = '''insert into ven_tw_sem_units (tw_id, cls_json_str, nps_str) values (?, ?, ?)'''
        timer_start = time.time()
        cnt = 0
        for (dirpath, dirname, filenames) in walk(sem_unit_folder):
            for filename in filenames:
                if filename[:14] == 'sem_units_int_' and filename[-5:] == '.json':
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        sem_units_json = json.load(in_fd)
                        in_fd.close()
                        for tw_id in sem_units_json:
                            cls_json_str = sem_units_json[tw_id]['cls']
                            nps_str = sem_units_json[tw_id]['nps']
                            try:
                                db_cur.execute(sql_str, (tw_id, cls_json_str, nps_str))
                            except Exception as err:
                                logging.debug('[SemUnitsExtractor:output_sem_units_to_db] %s' % err)
                                pass
                            cnt += 1
                            if cnt % 10000 == 0 and cnt >= 10000:
                                db_conn.commit()
                                logging.debug(
                                    '[SemUnitsExtractor:output_sem_units_to_db] %s sem units are written in %s secs.'
                                    % (cnt, time.time() - timer_start))
        db_conn.commit()
        logging.debug('[SemUnitsExtractor:output_sem_units_to_db] %s sem units are written in %s secs.'
                      % (cnt, time.time() - timer_start))
        db_conn.close()
        logging.debug('[SemUnitsExtractor:output_sem_units_to_db] All done in %s secs.'
                      % str(time.time() - timer_start))

    def sem_unit_stats(self, sem_unit_folder, output_folder):
        l_degrees = []
        l_nodes = []
        l_edges = []
        l_nps = []
        cnt = 0
        for (dirpath, dirname, filenames) in walk(sem_unit_folder):
            for filename in filenames:
                if filename[-14:] == '_cls_graph.gml':
                    sem_unit_name = filename[:-14]
                    try:
                        cls_graph = nx.read_gml(dirpath + '/' + filename)
                        l_nodes.append(cls_graph.number_of_nodes())
                        l_edges.append(cls_graph.number_of_edges())
                        l_degrees.append(sum(dict(cls_graph.degree()).values()) / float(cls_graph.number_of_nodes()))
                    except:
                        pass
                elif filename[-8:] == '_nps.txt':
                    sem_unit_name = filename[:-8]
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        lns = in_fd.readlines()
                        ln_cnt = len(lns)
                        l_nps.append(ln_cnt)
                        in_fd.close()
                cnt += 1
                if cnt % 10000 == 0 and cnt >= 10000:
                    with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
                        out_fd.write('cls: avg. degrees:\n')
                        out_fd.write(','.join([str(num) for num in l_degrees]))
                        out_fd.write('\n')
                        out_fd.write('cls: nodes:\n')
                        out_fd.write(','.join([str(num) for num in l_nodes]))
                        out_fd.write('\n')
                        out_fd.write('cls: edges:\n')
                        out_fd.write(','.join([str(num) for num in l_edges]))
                        out_fd.write('\n')
                        out_fd.write('nps:\n')
                        out_fd.write(','.join([str(num) for num in l_nps]))
                        out_fd.write('\n')
                        out_fd.close()
                    logging.debug('%s sem unit stats are done.' % cnt)
        with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
            out_fd.write('cls: avg. degrees:\n')
            out_fd.write(','.join([str(num) for num in l_degrees]))
            out_fd.write('\n')
            out_fd.write('cls: nodes:\n')
            out_fd.write(','.join([str(num) for num in l_nodes]))
            out_fd.write('\n')
            out_fd.write('cls: edges:\n')
            out_fd.write(','.join([str(num) for num in l_edges]))
            out_fd.write('\n')
            out_fd.write('nps:\n')
            out_fd.write(','.join([str(num) for num in l_nps]))
            out_fd.write('\n')
            out_fd.close()
