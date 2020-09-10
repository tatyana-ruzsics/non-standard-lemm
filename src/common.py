#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
#import codecs

# Default paths
SRC_FOLDER = os.path.dirname(__file__)
RESULTS_FOLDER = os.path.join(SRC_FOLDER, '../results')
DATA_FOLDER = os.path.join(SRC_FOLDER, '../data/')


# Model defaults
BEGIN_CHAR   = u'<s>'
STOP_CHAR   = u'</s>'
UNK_CHAR = u'<unk>'
BOUNDARY_CHAR = u' '

### IO handling and evaluation

def check_path(path, arg_name, is_data_path=True): #common
    if not os.path.exists(path):
        prefix = DATA_FOLDER if is_data_path else RESULTS_FOLDER
        tmp = os.path.join(prefix, path)
        if os.path.exists(tmp):
            path = tmp
        else:
            if is_data_path:
                print('%s incorrect: %s and %s' % (arg_name, path, tmp))
                raise ValueError
            else: #results path
                print(tmp)
                os.makedirs(tmp)
                path = tmp
    return path

def check_vocab_path(path, model_folder):
    if os.path.exists(path):
        vocab_path = path # absolute path  to existing vocab file
    else:
        tmp = os.path.join(RESULTS_FOLDER, path)
        if os.path.exists(tmp): # relative path to existing vocab file
            vocab_path = tmp
        else:
            vocab_path = os.path.join(model_folder,path) # no vocab - use default name
    return vocab_path

def write_pred_file(output_file_path, final_results, format = 0):
    
    print('len of predictions is {}'.format(len(final_results)))
    if format == 0:
        predictions_path = output_file_path + '.predictions'
        with open(predictions_path, 'w', encoding='utf8') as predictions:
            #for input, prediction in final_results:
            for res in final_results:
                predictions.write(u'\t'.join(res)+'\n')
                #predictions.write(u'{}\t{}\n'.format(input, prediction))
    elif format == 1:
        id = 0
        predictions_path = output_file_path + '.predictions'
        with open(predictions_path, 'w', encoding='utf8') as predictions:
            for input, beam_predictions  in final_results:
                for beam_prediction in beam_predictions:
                    nmt_score,lm_scores,prediction, weighted_score = beam_prediction
                    predictions.write(u'{} ||| {} ||| {} ||| {}\n'.format(id, prediction, u' '.join([str(-nmt_score)] + [str(-s) for s in lm_scores]), -weighted_score))
                id +=1

    return

def write_param_file(output_file_path, hyper_params):
    
    with open(output_file_path, 'w', encoding='utf8') as f:
        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')
    
    return

def write_eval_file(output_file_path, results, test_file_path):
    
    f = open(output_file_path + '.eval', 'w', encoding='utf8')
    f.write('File path = ' + str(test_file_path) + '\n')
    for measure, result in results.items():
        f.write('{} = {}\n'.format(measure, result))
    #f.write('{} = {}\n'.format(measure, result))
    
    return


# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}
    
    def save(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as fh:
            for w,i in sorted(self.w2i.items(),key=lambda v:v[0]):
                fh.write(u'{}\t{}\n'.format(w,i))
        return

    
    @classmethod
    def from_list(cls, words, w2i_=None):
        if w2i_:
            idx=len(w2i_)
            w2i=w2i_.copy()
        else:
            w2i = {}
            idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)
    
    @classmethod
    def from_file(cls, vocab_fname):
        w2i = {}
        with open(vocab_fname, 'r', encoding='utf-8') as fh:
            for line in fh:
                word, idx = line.rstrip().split('\t')
                w2i[word] = int(idx)
                #print word, idx
        return Vocab(w2i)
    
    def size(self): return len(self.w2i.keys())

def build_vocabulary(train_data, vocab_path, special_symbols_list=[], over_words=False, min_freq=0):
    # Build vocabulary over items - chars or segments - and save it to 'vocab_path'
    
    if not over_words:
        if min_freq > 0:
            char_tokens = [c for w in train_data for c in w if c not in special_symbols_list]
            char_counter = collections.Counter(char_tokens)
            char_counter_singletons = set([c for c in char_counter.keys() if char_counter[c]<=min_freq])
            print('found {} signleton characters: {}'.format(len(char_counter_singletons), ', '.join(list(char_counter_singletons))))
            items = sorted([c for c in char_counter.keys() if c not in char_counter_singletons])
        else:
            items = sorted(list(set([c for w in train_data for c in w if c not in special_symbols_list])))
            
        # items = sorted(list(set([c for w in train_data for c in w if c not in special_symbols_list])))
    else:
        items = sorted(list(set(train_data)))

    # to make sure that special symbols have the same index across models
    # treat special symbols first
    w2i = {}
    i=0
    if len(special_symbols_list)!=0:
        for c in special_symbols_list:
            w2i[c]=i
            i+=1

    print('special_symbols_list: {}'.format(special_symbols_list))
    print('Vocabulary size: {}'.format(len(items)+len(special_symbols_list)))
    print()
    vocab = Vocab.from_list(items,w2i)
    vocab.save(vocab_path)
    return
