#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import json
import argparse
from random import seed
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

import torch

from torch.utils.data import DataLoader

from collections import defaultdict
from encoders import *
from quantizers import *
from train import *
from utils.data import *
from utils.loss import *
from utils.summary import RougeEvaluator,truncate_summary_meantokenv3
from sklearn.cluster import AgglomerativeClustering
import transformers
from transformers import AutoModel, AutoTokenizer
import spacy_alignments as tokenizations
import numpy as np
import spacy
import copy
# parts of the code has been
# adapted from: https://github.com/stangelid/qt

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extracts aspect summaries with a trained SemAE model.\n')

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--summary_data',
                                help='summarization benchmark data',
                                type=str,
                                default='../data/space/json/space_summ.json')
    data_arg_group.add_argument('--gold_data',
                                help='gold data root directory',
                                type=str,
                                default='../data/space/gold')
    data_arg_group.add_argument(
        '--gold_aspects',
        help=
        'aspect categories to evaluate against (default: all SPACE aspects)',
        type=str,
        default='building,cleanliness,food,location,rooms,service')
    argparser.add_argument(
        '--seedsdir',
        help='directory that holds aspect query words, i.e., seeds',
        type=str,
        default='../data/seeds')
    argparser.add_argument('--max_num_seeds',
                           help='number of seed words per aspect',
                           type=int,
                           default=5)
    data_arg_group.add_argument(
        '--sentencepiece',
        help='sentencepiece model file',
        type=str,
        default='../data/sentencepiece/spm_unigram_32k.model')
    data_arg_group.add_argument(
        '--max_rev_len',
        help='maximum number of sentences per review (default: 150)',
        type=int,
        default=150)
    data_arg_group.add_argument(
        '--max_sen_len',
        help='maximum number of tokens per sentence (default: 40)',
        type=int,
        default=40)
    data_arg_group.add_argument(
        '--split_by',
        help='how to split summary data (use "alphanum" for SPACE)',
        type=str,
        default='alphanum')

    summ_arg_group = argparser.add_argument_group('Summarizer arguments')
    summ_arg_group.add_argument('--model',
                                help='trained QT model to use',
                                type=str,
                                default='../models/run11_20_model.pt')
    summ_arg_group.add_argument(
        '--manual_head',
        help=
        'manually set aspect head for extraction (default: auto via entropy)',
        type=int,
        default=None)
    summ_arg_group.add_argument(
        '--truncate_clusters',
        help=
        'truncate cluster sampling to top-p % of clusters (if < 1) or top-k (if > 1)',
        type=float,
        default=1.0)
    summ_arg_group.add_argument('--beta',
                                help='Parameter beta',
                                type=float,
                                default=0.7)
    summ_arg_group.add_argument(
        '--num_cluster_samples',
        help='number of cluster samples (default: 300)',
        type=int,
        default=300)
    summ_arg_group.add_argument(
        '--sample_sentences',
        help=
        'enable 2-step sampling (sample sentences within cluster neighbourhood)',
        action='store_true')
    summ_arg_group.add_argument(
        '--truncate_cluster_nn',
        help=
        'truncate sentences that live in a cluster neighborhood (default: 5)',
        type=int,
        default=5)
    summ_arg_group.add_argument(
        '--num_sent_samples',
        help='number of sentence samples per cluster sample (default: 30)',
        type=int,
        default=30)
    summ_arg_group.add_argument(
        '--temp',
        help='temperature for sampling sentences within cluster (default: 10)',
        type=int,
        default=3)

    out_arg_group = argparser.add_argument_group('Output control')
    out_arg_group.add_argument('--outdir',
                               help='directory to put summaries',
                               type=str,
                               default='../outputs')
    out_arg_group.add_argument('--max_tokens',
                               help='summary budget in words (default: 40)',
                               type=int,
                               default=40)
    out_arg_group.add_argument(
        '--min_tokens',
        help='minimum summary sentence length in words (default: 1)',
        type=int,
        default=1)
    out_arg_group.add_argument(
        '--cos_thres',
        help='cosine similarity threshold for extraction (default: 1.0)',
        type=float,
        default=1.0)
    out_arg_group.add_argument('--no_cut_sents',
                               help='don\'t cut last summary sentence',
                               action='store_true')
    out_arg_group.add_argument('--no_early_stop',
                               help='allow last sentence to go over limit',
                               action='store_true')
    out_arg_group.add_argument(
        '--newline_sentence_split',
        help='one sentence per line (don\'t use if evaluating with ROUGE)',
        action='store_true')
    out_arg_group.add_argument(
        '--num_cluster',
        help='number of aspect-related words clusters',
        type=int,
        default=6)
    out_arg_group.add_argument(
        '--gamma',
        help='number of aspect-related words clusters',
        type=float,
        default=0.6)
    out_arg_group.add_argument(
        '--beta1',
        help='number of aspect-related words clusters',
        type=float,
        default=5e-3)

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument('--run_id',
                                 help='unique run id (for outputs)',
                                 type=str,
                                 default='aspect_run_new')
    other_arg_group.add_argument('--no_eval',
                                 help='don\'t evaluate (just write summaries)',
                                 action='store_true')
    other_arg_group.add_argument(
        '--gpu',
        help='gpu device to use (default: -1, i.e., use cpu)',
        type=int,
        default=1)
    other_arg_group.add_argument('--batch_size',
                                 help='the maximum batch size (default: 5)',
                                 type=int,
                                 default=5)
    other_arg_group.add_argument('--sfp',
                                 help='system filename pattern for pyrouge',
                                 type=str,
                                 default='(.*)')
    other_arg_group.add_argument('--mfp',
                                 help='model filename pattern for pyrouge',
                                 type=str,
                                 default='#ID#_[012].txt')
    other_arg_group.add_argument('--seed',
                                 help='random seed',
                                 type=int,
                                 default=1)
    args = argparser.parse_args()

    seed(1)

    device = torch.device('cuda:{0}'.format(args.gpu))

    # set paths
    summ_data_path = args.summary_data
    model_path = args.model
    output_path = os.path.join(args.outdir, args.run_id)
    eval_path = args.outdir
    gold_path = args.gold_data
    seeds_path = args.seedsdir
    spm_path = args.sentencepiece
    beta = args.beta
    spacy_model=spacy.load('en_core_web_sm')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model_bert.eval()
    assert args.model != '', 'Please give model path'

    # read aspect seed words
    aspects = args.gold_aspects.split(',')
    num_aspects = len(aspects)
    aspect_indices = {}
    aspect_seeds = {}

    for i, aspect in enumerate(aspects):
        aspect_indices[aspect] = i
        seeds = {}
        f = open(os.path.join(seeds_path, aspect + '.txt'), 'r')
        count=0
        for line in f:
            conf, seed_word = line.split()
            seeds[seed_word] = conf
            count+=1
            if count>=args.max_num_seeds:
                break
        f.close()
        aspect_seeds[aspect] = seeds

    # aspect mapping tools
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # load summarization data
    f = open(summ_data_path, 'r')
    summ_data = json.load(f)
    f.close()

    # prepare summarization dataset
    summ_dataset = ReviewSummarizationDataset(summ_data,
                                              spmodel=spm_path,
                                              max_rev_len=args.max_rev_len,
                                              max_sen_len=args.max_sen_len)
    vocab_size = summ_dataset.vocab_size
    pad_id = summ_dataset.pad_id()
    bos_id = summ_dataset.bos_id()
    eos_id = summ_dataset.eos_id()
    unk_id = summ_dataset.unk_id()

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=pad_id,
                              unk_idx=unk_id,
                              bos_idx=bos_id,
                              eos_idx=eos_id)

    # split dev/test entities
    summ_dataset.entity_split(split_by=args.split_by)

    # create entity data loaders
    summ_dls = {}
    summ_samplers = summ_dataset.get_entity_batch_samplers(args.batch_size)
    for entity_id, entity_sampler in summ_samplers.items():
        summ_dls[entity_id] = DataLoader(
            summ_dataset,
            batch_sampler=entity_sampler,
            collate_fn=collator.collate_reviews_with_ids)

    torch.manual_seed(args.seed)

    # Model Loading

    model = torch.load(args.model, map_location=device)
    
    nheads = model.encoder.output_nheads
    codebook_size = model.codebook_size
    d_model = model.d_model
    model.eval()

    # Prepare Aspect data
    def get_aspects(sent):
        results = {}
        for aspect in aspects:
            for x in aspect_seeds[aspect].keys():
                if x in sent:
                    results[aspect_seeds[aspect][x]] = aspect

        if not results:
            return ["others"]
        return results.values()

    all_texts = []
    ranked_entity_sentences = defaultdict(dict)
    aspect_text_dict={}
    num_cluster=args.num_cluster
    all_dist={}
    with torch.no_grad():
        for entity_id, entity_loader in tqdm(summ_dls.items()):
            texts = []
            distances = []
            noun_distances=[]
            bert_repres=[]
            for batch in entity_loader:
                src = batch[0].to(device)
                ids = batch[2]
                batch_txt=[]
                for full_id in ids:
                    entity_id, review_id = full_id.split('__')
                    batch_txt.extend(summ_dataset.reviews[entity_id][review_id])
                batch_size, nsent, ntokens = src.size()

                _, _, _, dist = model.cluster(src)
                inputs=tokenizer(batch_txt,padding=True, truncation=True, return_tensors="pt")
                bert_repre=[tokenizer.convert_ids_to_tokens(list(inputs['input_ids'][j].numpy()),skip_special_tokens=True) for j in range(inputs['input_ids'].size(0))]
                bert_repres.extend(bert_repre)
                inputs={k:inputs[k].to(device) for k in inputs.keys()}
                texts.extend(batch_txt)
                hidden_states=list(model_bert(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state)
                distances.extend(dist)
                noun_distances.extend(hidden_states)
            output1 = torch.stack(distances)
            sent_list1=texts
            assert len(sent_list1)==output1.shape[0]
            spacy_repres=list(spacy_model.pipe(texts))
            select_idx=[]
            noun_dist_list=[]
            assert len(bert_repres)==len(noun_distances)
            noun_word_list=[]
            noun_emb_list=[]
            noun_idx_list=[]
            noun_sent_list=[]
            for i in range(len(spacy_repres)):
                spacy_sents=list(spacy_repres[i].sents)
                docs=spacy_repres[i]
                spacy_token=[]
                spacy_tag=[]
                for k in spacy_sents:
                  spacy_token.extend([str(w) for w in k])
                  spacy_tag.extend([w.tag_ for w in k])
                spacy_tag_idx=[]
                noun_chunk_id=[]
                for k in docs.noun_chunks:
                    if k.root.tag_ in ['NN','NNS']:
                        spacy_tag_idx.append(k.root.i)
                    for w in k:
                        if w.tag_ in ['JJ','JJR','JJS']:
                            noun_chunk_id.append(w.i)
                spacy_tag_idx.extend([k for k in range(len(spacy_tag)) if spacy_tag[k] in ['JJ','JJR','JJS'] and k not in noun_chunk_id])
                a2b, b2a = tokenizations.get_alignments(bert_repres[i], spacy_token)
                if len(spacy_tag_idx)>0:
                    select_idx.append(i)
                start_idx=len(noun_idx_list)
                for k in range(len(b2a)):
                    if k in spacy_tag_idx and len(b2a[k])>0:
                        noun_word_list.append(spacy_token[k])
                        word_idx=[j+1 for j in b2a[k]]
                        noun_emb_list.append(torch.mean(noun_distances[i][word_idx],dim=0))
                        noun_idx_list.append(i)
                noun_sent_list.append(list(range(start_idx,len(noun_idx_list))))
            assert len(noun_emb_list)==len(noun_word_list)
            aspect_wise_sentences = defaultdict(list)
            for i, sentence in enumerate(texts):
                tokens = [
                    tok for tok in token_pattern.findall(sentence.lower())
                    if tok not in stop_words
                ]
                tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
                sent_aspects = list(get_aspects(tokens))

                if len(sent_aspects) == 0:
                    continue

                for aspect in sent_aspects:
                    aspect_wise_sentences[aspect].append(i)
            del(aspect_wise_sentences['others'])
            aspect_wise_texts={}
            for aspect in aspect_wise_sentences.keys():
                aspect_wise_texts[aspect]=[texts[i] for i in aspect_wise_sentences[aspect]]
            aspect_text_dict[entity_id]=aspect_wise_texts
            for aspect in aspect_wise_sentences.keys():
                aspect_dist = []
                aspect_text=copy.deepcopy(aspect_wise_texts[aspect])
                aspect_noun=[]
                for idx in aspect_wise_sentences[aspect]:
                    aspect_dist.append(distances[idx])
                    aspect_noun.append(noun_sent_list[idx])
                aspect_dist = torch.stack(aspect_dist)
                media_aspect_noun=copy.deepcopy(aspect_noun[0])
                for j in aspect_noun[1:]:
                    media_aspect_noun+=copy.deepcopy(j)
                aspect_noun_word_list=[noun_word_list[i] for i in media_aspect_noun]
                aspect_noun_emb_list=[noun_emb_list[i] for i in media_aspect_noun]
                aspect_noun_idx_list=[]
                for j in range(len(aspect_noun)):
                    aspect_noun_idx_list.extend([j]*len(aspect_noun[j]))
                seeds_list=None
                if aspect!='others':
                    seeds_list=[i for i in aspect_seeds[aspect].keys()]
                    seed_idx_list=[]
                    for i in aspect_noun_word_list:
                        if lemmatizer.lemmatize(i.lower()) in seeds_list:
                            seed_idx_list.append(True)
                        else:
                            seed_idx_list.append(False)
                    seed_idx_list=np.array(seed_idx_list)
                aspect_noun_emb_list=torch.stack(aspect_noun_emb_list)
                if seeds_list is not None:
                    mean_noun=torch.mean(aspect_noun_emb_list[seed_idx_list],dim=0,keepdim=True)
                else:
                    mean_noun=torch.mean(aspect_noun_emb_list,dim=0,keepdim=True)
                dist_matrix=torch.sum((aspect_noun_emb_list-mean_noun)**2,dim=1)
                select_indices=torch.argsort(dist_matrix)
                select_indices=select_indices[0:int(select_indices.size(0)*args.gamma)]
                aspect_noun_emb_list=aspect_noun_emb_list[select_indices]
                aspect_noun_idx_list=[aspect_noun_idx_list[i.item()] for i in select_indices]
                aspect_noun_word_list=[aspect_noun_word_list[i.item()] for i in select_indices]
                kmeans=AgglomerativeClustering(n_clusters=num_cluster).fit(aspect_noun_emb_list.detach().cpu().numpy())
                clusters=[[] for j in range(num_cluster)]
                clusters_word=[[] for j in range(num_cluster)]
                clusters_emb=[[] for j in range(num_cluster)]
                noun_sent_dict={i:[] for i in range(aspect_dist.size(0))}
                for j in range(len(kmeans.labels_)):
                    clusters[kmeans.labels_[j]].append(aspect_noun_idx_list[j])
                    clusters_word[kmeans.labels_[j]].append(aspect_noun_word_list[j])
                    clusters_emb[kmeans.labels_[j]].append(aspect_noun_emb_list[j])
                    noun_sent_dict[aspect_noun_idx_list[j]].append((aspect_noun_word_list[j],kmeans.labels_[j]))
                for j in range(num_cluster):
                    clusters[j]=list(set(clusters[j]))
                for j in range(len(clusters_word)):
                    clusters_emb[j]=torch.stack(clusters_emb[j])
                    media_center=torch.mean(clusters_emb[j],dim=0,keepdim=True)
                    media_dist=torch.sum((clusters_emb[j]-media_center)**2,dim=1)
                    media_idx=torch.argsort(media_dist)
                    media_list=[]
                    for i in media_idx:
                        if clusters_word[j][i.item()].lower() not in media_list:
                            media_list.append(clusters_word[j][i.item()].lower())
                        if len(media_list)>5:
                            break
                ranked_idx=[]
                sent_dist=9999*np.ones((aspect_dist.size(0),num_cluster))
                all_idx=np.arange(aspect_dist.size(0))
                in_cluster=[]
                for j in range(len(clusters)):
                    media_outputs=aspect_dist[clusters[j]]
                    media_idx=all_idx[clusters[j]]
                    media_center=torch.mean(media_outputs,dim=0)
                    in_cluster.append(media_outputs.size(0))
                    for k in range(media_outputs.size(0)):
                        D_z = media_outputs[k]
                        sent_dist[media_idx[k],j]=(kl_div_all_heads(D_z,media_center)).item()
                        assert sent_dist[media_idx[k],j]<9999
                in_cluster=np.array(in_cluster)
                sent_dist=sent_dist[:,in_cluster>2]
                in_cluster=in_cluster[in_cluster>2]
                in_cluster_idx=np.argsort(-in_cluster)
                sent_dist=sent_dist[:,in_cluster_idx]
                in_cluster=in_cluster[in_cluster_idx]
                assert len(aspect_text)==sent_dist.shape[0]
                ranked_idx=np.argsort(np.min(sent_dist,axis=1))
                sent_dist=sent_dist[ranked_idx]
                ranked_sentence_texts=[aspect_text[i] for i in ranked_idx]
                noun_sent_dict=[noun_sent_dict[i] for i in ranked_idx]
                len_list=[float(len(ranked_sentence_texts[i].split()))/float(np.sum(sent_dist[i]!=9999)) if float(np.sum(sent_dist[i]!=9999))!=0.0 else 1.0 for i in range(len(ranked_sentence_texts)) ]
                len_list=np.log(np.array(len_list))
                len_list=args.beta1*np.expand_dims(len_list,axis=1)
                assert len_list.shape[0]==len(ranked_sentence_texts)
                assert len_list.shape[0]==sent_dist.shape[0]
                all_dist[entity_id+aspect]=sent_dist
                sent_dist=sent_dist+len_list
                ranked_entity_sentences[entity_id][
                    aspect] = ranked_sentence_texts
                ranked_entity_sentences[entity_id][aspect] = (ranked_sentence_texts,sent_dist,noun_sent_dict)
            all_texts.extend(texts)
    # tfidf vectorizer used for cosine threshold
    if args.cos_thres != -1:
        vectorizer = TfidfVectorizer(decode_error='replace',
                                     stop_words='english')
        vectorizer.fit(all_texts)
    else:
        vectorizer = None
    # write summaries
    dict_results = {'dev': {}, 'test': {}, 'all': {}}
    all_outputs = []

    if args.newline_sentence_split:
        delim = '\n'
    else:
        delim = '\t'
    for aspect in tqdm(aspects):
        aspect_output_path = os.path.join(output_path, aspect)
        os.makedirs(aspect_output_path, exist_ok=True)

        for entity_id in ranked_entity_sentences:
            if entity_id in summ_dataset.dev_entity_ids:
                file_path = os.path.join(aspect_output_path,
                                         'dev_' + entity_id)
            else:
                file_path = os.path.join(aspect_output_path,
                                         'test_' + entity_id)

            ranked_sentences = ranked_entity_sentences[entity_id][aspect]
            summary_sentences = truncate_summary_meantokenv3(
                ranked_sentences[0],
                ranked_sentences[1],
                max_tokens=args.max_tokens,
                max_tokens_sent=args.max_tokens,
                cut_sents=(not args.no_cut_sents),
                vectorizer=vectorizer,
                cosine_threshold=args.cos_thres,
                early_stop=(not args.no_early_stop),
                noun_sent_dict=ranked_sentences[2],
                min_tokens=args.min_tokens)

            fout = open(file_path, 'w')
            fout.write(delim.join(summary_sentences))
            fout.close()
        # evaluate summaries
        model_dir = os.path.join(gold_path, aspect)
        dev_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                       model_dir=model_dir,
                                       system_filename_pattern='dev_' +
                                       args.sfp,
                                       model_filename_pattern=args.mfp)
        test_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                        model_dir=model_dir,
                                        system_filename_pattern='test_' +
                                        args.sfp,
                                        model_filename_pattern=args.mfp)
        all_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                       model_dir=model_dir,
                                       system_filename_pattern='[^_]*_' +
                                       args.sfp,
                                       model_filename_pattern=args.mfp)

        outputs = dev_evaluator.evaluate()
        dict_results['dev'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [dev]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = test_evaluator.evaluate()
        dict_results['test'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [test]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = all_evaluator.evaluate()
        dict_results['all'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [all]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

    ftxt = open(os.path.join(eval_path, 'eval_{0}.txt'.format(args.run_id)),
                'w')
    ftxt.write('\n'.join(all_outputs))
    ftxt.close()

    fjson = open(os.path.join(eval_path, 'eval_{0}.json'.format(args.run_id)),
                 'w')
    fjson.write(json.dumps(dict_results))
    fjson.close()