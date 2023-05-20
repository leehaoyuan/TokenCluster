from pyrouge import Rouge155
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
import logging
#from rouge import Rouge
import os
import scipy
import numpy as np
import copy
PUNCT = set(string.punctuation)


# parts of the code has been
# adapted from: https://github.com/stangelid/qt

def truncate_summary(ranked_sentences,
                     ranked_idx=None,
                     max_tokens=75,
                     min_tokens=1,
                     cut_sents=False,
                     early_stop=True,
                     remove_non_alpha=True,
                     vectorizer=None,
                     cosine_threshold=None):
    '''Truncates a summary by iteratively adding sentences
       until the max_tokens limit is passed. 
    '''
    count = 0
    summary = []
    summary_sentence_ids = []
    summ_ids=[]
    def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set
    def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False
    if vectorizer is not None:
        assert cosine_threshold > 0 and cosine_threshold <= 1, \
                'cosine threshold should be in (0,1]'
        sentence_vecs = vectorizer.transform(ranked_sentences)
        similarities = cosine_similarity(sentence_vecs)

    for i, sentence in enumerate(ranked_sentences):
        if remove_non_alpha and all(c.isdigit() or c in PUNCT
                                    for c in sentence):
            continue

        if len(sentence.split()) < min_tokens:
            continue
        if len(sentence.split()) > max_tokens:
            continue
        if vectorizer is not None and i > 0:
            
            similarities_to_existing = similarities[i, summary_sentence_ids]
            if not all(similarities_to_existing < cosine_threshold): 
                continue
        summary.append(sentence)
        summary_sentence_ids.append(i)
        if ranked_idx is not None:
            summ_ids.append(ranked_idx[i])
        count += len(sentence.split())
        if count > max_tokens:
            if cut_sents:
                last_sent = summary[-1].split()
                last_sent = last_sent[:len(last_sent) - count + max_tokens]
                if len(last_sent) > 0:
                    summary[-1] = ' '.join(last_sent)
                else:
                    summary = summary[:-1]
                break
            else:
                summary = summary[:-1]
                if early_stop:
                    break
                else:
                    count -= len(sentence.split())
                    summary_sentence_ids = summary_sentence_ids[:-1]
    if ranked_idx is None:
        return summary
    else:
        return summary,summ_ids
def truncate_summary_meantokenv3(ranked_sentences,
          ranked_dist,
          ranked_ids=None,
          max_tokens=75,
          min_tokens=1,
          max_tokens_sent=999,
          cut_sents=False,
          early_stop=True,
          remove_non_alpha=True,
          vectorizer=None,
          noun_sent_dict=None,
          cosine_threshold=None):
    '''Truncates a summary by iteratively adding sentences
       until the max_tokens limit is passed. 
    '''
    count = 0
    summary = []
    summ_vecs=None
    summ_ids=[]
    dist_matrix=copy.deepcopy(ranked_dist)
    no_max=True
    round_group_idx=np.zeros(ranked_dist.shape[1])
    previous_idx=None
    summ_aspects=[]
    while no_max:
        if np.min(dist_matrix[:,round_group_idx==0])<1000:
            ranked_idx=np.argsort(np.min(dist_matrix[:,round_group_idx==0],axis=1))
            cur_sent_idx=None
            if cur_sent_idx is None:
                cur_sent_idx=ranked_idx[0]
                if previous_idx is not None and cur_sent_idx==previous_idx:
                    assert 0==1
                previous_idx=cur_sent_idx
            if remove_non_alpha and all(c.isdigit() or c in PUNCT for c in ranked_sentences[cur_sent_idx]):
              dist_matrix[cur_sent_idx]=9999
              continue
            if len(ranked_sentences[cur_sent_idx].split())<min_tokens:
              dist_matrix[cur_sent_idx]=9999
              continue
            if len(ranked_sentences[cur_sent_idx].split()) > max_tokens:
              dist_matrix[cur_sent_idx]=9999
              continue
            if vectorizer is not None and len(summary)>0:
              sentence_vecs=vectorizer.transform([ranked_sentences[cur_sent_idx]])
              media_aspect=np.vstack(summ_aspects)<1000
              media_aspect1=ranked_dist[cur_sent_idx:(cur_sent_idx+1)]<1000
              media_aspect=media_aspect*media_aspect1
              media_aspect=np.sum(media_aspect,axis=1)>0
              media_summ_vecs=summ_vecs[media_aspect]
              if media_summ_vecs.shape[0]>=1:
                  similarities_to_existing=cosine_similarity(media_summ_vecs,sentence_vecs)[:,0]
                  if not all(similarities_to_existing < cosine_threshold):
                    dist_matrix[cur_sent_idx]=9999
                    continue
            assert np.min(dist_matrix[cur_sent_idx])<1000
            summary.append(ranked_sentences[cur_sent_idx])
            summ_aspects.append(ranked_dist[cur_sent_idx])
            if ranked_ids is not None:
              summ_ids.append(ranked_ids[cur_sent_idx])
            if summ_vecs is None:
              summ_vecs=vectorizer.transform([ranked_sentences[cur_sent_idx]])
            else:
              media=vectorizer.transform([ranked_sentences[cur_sent_idx]])
              summ_vecs=scipy.sparse.vstack((summ_vecs,media))
            count+=len(ranked_sentences[cur_sent_idx].split())
            if count> max_tokens:
              if cut_sents:
                last_sent = summary[-1].split()
                last_sent = last_sent[:len(last_sent) - count + max_tokens]
                if len(last_sent) > 0:
                    summary[-1] = ' '.join(last_sent)
                else:
                    summary = summary[:-1]
                    summ_aspects=summ_aspects[:-1]
                no_max=False
                break
              else:
                summary=summary[:-1]
                summ_aspects=summ_aspects[:-1]
                if ranked_ids is not None:
                  summ_ids=summ_ids[:-1]
                if early_stop:
                  no_max=False
                  break
                else:
                  count-=len(ranked_sentences[cur_sent_idx].split())
                  summ_vecs=summ_vecs[:-1]
            else:
              dist_matrix[cur_sent_idx]=9999
            for i in range(ranked_dist[cur_sent_idx].shape[0]):
              if ranked_dist[cur_sent_idx][i]<1000:
                  round_group_idx[i]=1
        else:
            round_group_idx=1
        if np.sum(round_group_idx==0)==0:
            round_group_idx=np.zeros(ranked_dist.shape[1])
            for i in range(ranked_dist.shape[1]):
                if np.min(dist_matrix[:,i])>=1000:
                    round_group_idx[i]=1
        if np.sum(round_group_idx==0)==0:
            break
    assert len(summary)==len(summ_aspects)
    summ_aspects=np.vstack(summ_aspects)<1000
    summ_aspects=summ_aspects.astype(float)
    remain=np.zeros(len(summary))
    remain[0]=1.0
    finals=[0]
    similarities=cosine_similarity(summ_aspects)
    for j in range(len(summary)-1):
        cur_idx=finals[-1]
        rank_idx=np.argsort(-similarities[cur_idx])
        for k in rank_idx:
            if remain[k]==0.0:
                remain[k]=1.0
                finals.append(k)
                break
    if ranked_ids is None:
      return summary
    else:
      return summary,summ_ids


class RougeEvaluator():
    '''Wrapper for pyrouge'''
    def __init__(self,
                 system_filename_pattern='([0-9]*)',
                 model_filename_pattern='#ID#_[012].txt',
                 system_dir=None,
                 model_dir=None,
                 log_level=logging.WARNING):
        self.system_dir = system_dir
        self.model_dir = model_dir

        self.r = Rouge155()
        self.r.log.setLevel(log_level)
        self.r.system_filename_pattern = system_filename_pattern
        self.r.model_filename_pattern = model_filename_pattern

        self.results_regex = \
                re.compile('(ROUGE-[12L]) Average_F: ([0-9.]*) \(95.*?([0-9.]*) - ([0-9.]*)')

    def evaluate(self, system_dir=None, model_dir=None):
        if system_dir is None:
            assert self.system_dir is not None, 'no system_dir given'
            system_dir = self.system_dir
        if model_dir is None:
            assert self.model_dir is not None, 'no model_dir given'
            model_dir = self.model_dir

        self.r.system_dir = system_dir
        self.r.model_dir = model_dir

        full_output = self.r.convert_and_evaluate()
        results = self.results_regex.findall(full_output)

        outputs = {}
        outputs['full_output'] = full_output
        outputs['dict_output'] = self.r.output_to_dict(full_output)
        outputs['short_output'] = '\n'.join(
            ['  {0} {1} ({2} - {3})'.format(*r) for r in results])

        return outputs
def rouge_evaluate(self,system_dir,model_dir,prefix=None):
    hyp_list=os.listdir(system_dir)
    if prefix is not None:
        hyp_list=[i for i in hyp_list if i.startswith(prefix)]
    id_list=[i.split('_')[0] for i in hyp_list]
    pre_ref_list=os.listdir(model_dir)
    ref_dict=[]
    for i in id_list:
        media_list=[j for j in pre_ref_list if j.startswith(i+'_')]
        ref_dict.append(media_list)
    for i in range(len(hyp_list)):
        if len(ref_dict[i])>0:
            with open(os.path.join(system_dir,hyp_list[i]),'r') as file:
                for j in file:
                    hyp=j.replace('')
                
    
        
