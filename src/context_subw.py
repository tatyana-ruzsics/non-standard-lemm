#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft_sub_context.py train [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch NUM]
    [--char_input=CHAR_INPUT] [--tag_input=TAG_INPUT] [--hidden_enc=HIDDEN_ENC] [--batch_size=BATCH]
    [--hidden_context=HIDDEN_CONTEXT] [--enc_layers=ENC_LAYERS] [--char_vocab_path=VOCAB_PATH_CHAR] 
    [--tag_vocab_path=VOCAB_PATH_TAG] [--dec_layers=DEC_LAYERS] [--beam=BEAM] [--hidden_dec=HIDDEN_DEC]
    [--enc_cont_layers=ENC_CONT_LAYERS] [--report_freq=REPORT_FREQ] 
    [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION] 
    [--subw_context] [--word_context] [--word_context_gated]
    [--verbose] [--char_min_freq=MIN_FREQ] [--lower_target]
    MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE [--train_bpe_path=TRAIN_BPE_FILE] [--dev_bpe_path=DEV_BPE_FILE]
  norm_soft_sub_context.py test [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE] 
    [--verbose]
    MODEL_FOLDER --test_path=TEST_FILE [--test_bpe_path=TEST_BPE_FILE]
    

Arguments:
MODEL_FOLDER  save/read model folder where also eval results are written to, possibly relative to RESULTS_FOLDER
ED_MODEL_FOLDER  ED model(s) folder, possibly relative to RESULTS_FOLDER, coma-separated

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --dynet-autobatch NUM         turn on dynet autobatching [default: 1]
  --char_input=CHAR_INPUT       charachters input vector dimensions [default: 200]
  --tag_input=TAG_INPUT         tag input vector dimension [default: 200]
  --hidden_enc=HIDDEN_ENC          hidden layer dimensions for encoder LSTM [default: 100]
  --hidden_context=HIDDEN_CONTEXT  hidden layer dimensions for context LSTM [default: 100]
  --hidden_dec=HIDDEN_DEC          hidden layer dimensions for decoder LSTM [default: 100]
  --enc_layers=ENC_LAYERS       amount of layers in encoder LSTMs  [default: 2]
  --dec_layers=DEC_LAYERS       amount of layers in decoder LSTMs  [default: 2]
  --enc_cont_layers=ENC_CONT_LAYERS       amount of layers in decoder LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0]
  --epochs=EPOCHS               number of training epochs   [default: 50]
  --patience=PATIENCE           patience for early stopping [default: 5]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/AMSGrad/ADADELTA [default: AMSGrad]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --train_bpe_path=TRAIN_BPE_FILE       train subwords set path, possibly relative to DATA_FOLDER, only for training
  --dev_bpe_path=DEV_BPE_FILE           dev subwords set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --test_bpe_path=TEST_BPE_FILE          test subwords set path, possibly relative to DATA_FOLDER, only for training
  --char_vocab_path=VOCAB_PATH_CHAR  char vocab path, possibly relative to RESULTS_FOLDER [default: char_vocab.txt]
  --tag_vocab_path=VOCAB_PATH_TAG  tag vocab path, possibly relative to RESULTS_FOLDER [default: tag_vocab.txt]
  --beam=BEAM                   beam width [default: 1]
  --pred_path=PRED_FILE         name for predictions file in the test mode [default: best.test]
  --subw_context                use subwords hierarchical context (with gated attention) over chars/subwords [default: False]
  --word_context                use words hierarchical context [default: False]
  --word_context_gated          use words hierarchical context with gated attention over chars/subwords [default: False]    
  --batch_size=BATCH            batch size [default: 1]
  --char_min_freq=MIN_FREQ      remove characters from vocabulary with threshold frequency of less or equal MIN_FREQ [default: 0]
  --report_freq=REPORT_FREQ     after how many batches to preform evaluation [default: 200]
  --verbose                     verbose decoding
  --lower_target                lower target prediction [default: False]
"""

from __future__ import division
from docopt import docopt
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict

import time
import copy
import dynet as dy
import numpy as np
import os

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, \
                    SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER, \
                    check_path, check_vocab_path, write_pred_file, write_param_file, write_eval_file, \
                    build_vocabulary, Vocab

from tools.preprocess import extract_data_with_subw, extract_data

SPECIAL_CHARS=[BEGIN_CHAR,STOP_CHAR,UNK_CHAR]

MAX_PRED_SEQ_LEN = 50 # option
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, alpha=0.0001, #common
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
                'SGD'     : dy.SimpleSGDTrainer,
                'AMSGrad' : lambda m: dy.AmsgradTrainer(m),
                'ADADELTA': dy.AdadeltaTrainer}



def log_to_file(log_file_name, info):    
    print(info)
    with open(log_file_name, "a") as logfile:
        logfile.write(info+'\n')

class DataInstance(object):
    def __init__(self, input, output, subw_len=None, subw_pos=None, lower_target=False):
        self.input = input
        self.output = output.lower() if lower_target else output
        self.subw_len = subw_len
        self.subw_pos = subw_pos

    def encode(self, vocab):
        UNK = vocab.w2i[UNK_CHAR]

        input_padded = [BEGIN_CHAR] + [c for c in self.input] + [STOP_CHAR]
        self.input2ind = [vocab.w2i.get(c, UNK) for c in input_padded]
        if self.subw_len:   
            self.subw_len[0]+=1
            self.subw_len[-1]+=1

        output_padded = [c for c in self.output] + [STOP_CHAR]
        self.output2ind = [vocab.w2i.get(c, UNK) for c in output_padded]

class DataSet(object):
    def __init__(self, dataset, *args, **kwargs):
        print(dataset[1]) 
        # dataset: list of sents, where sent = (inputs, outputs)
        # dataset with subword splits: list of sents, where sent = (inputs, outputs, subw_lens, subw_positions)
        self.dataset = []
        for sent in dataset:
            sent2objects=[]
            for item in zip(*sent):
                # item = input, output, subw_len, subw_pos
                d_item = DataInstance(*item, *args, **kwargs)
                #import pdb; pdb.set_trace()
                sent2objects.append(d_item)
            self.dataset.append(sent2objects)
            sent2objects = []

        self.length = len(self.dataset)

    def encode(self, vocab):
        for sent in self.dataset:
            for item in sent:
                #print(item)
                item.encode(vocab)
            
    def iter(self, indices=None, shuffle=False, batch_size =1):
#        zipped = zip(*self.dataset)
        zipped = self.dataset
        if indices or shuffle:
            if not indices:
                indices = list(range(self.length))
            elif isinstance(indices, int):
                indices = list(range(indices))
            else:
                assert isinstance(indices, (list, tuple))
            if shuffle:
                random.shuffle(indices)
            zipped = [zipped[i] for i in indices]
        zipped = [zipped[i:i+batch_size] for i in range(0,len(zipped),batch_size)]
        return zipped

    def inputs(self):
        return [w.input for sent in self.dataset for w in sent]

    def outputs(self):
        return [w.output for sent in self.dataset for w in sent]
    
    @classmethod
    def from_file(cls, path, bpe_path, dataset_reader, *args, **kwargs):
        # returns a `SoftDataSet` with fields: inputs, outputs
        dataset = dataset_reader(path, bpe_path, *args, **kwargs)
        return cls(dataset, *args, **kwargs)


class SoftAttention(object):
    def __init__(self, model_hyperparams, best_model_path=None):

        self.add_params(model_hyperparams)
        if best_model_path:
            self.model.populate(best_model_path)

    def add_params(self, model_hyperparams):
        self.model=dy.Model()

        self.hyperparams = model_hyperparams
            
        self.char_vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH_CHAR'])
        self.BEGIN   = self.char_vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.char_vocab.w2i[STOP_CHAR]
        self.UNK    = self.char_vocab.w2i[UNK_CHAR] 
       # self.hyperparams['VOCAB_SIZE_CHAR'] = self.char_vocab.size()

        self.dim_enc = self.hyperparams['HIDDEN_DIM_ENC']
        self.dim = self.hyperparams['HIDDEN_DIM_DEC']
        # BiLSTM for input
        self.fbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_LAYERS'], self.hyperparams['INPUT_DIM_CHAR'], \
            self.dim_enc, self.model)
        self.bbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_LAYERS'], self.hyperparams['INPUT_DIM_CHAR'], \
            self.dim_enc, self.model)      
        # embedding lookups for vocabulary (chars)
        self.CHAR_VOCAB_LOOKUP  = self.model.add_lookup_parameters((self.hyperparams['VOCAB_SIZE_CHAR'], \
            self.hyperparams['INPUT_DIM_CHAR']))
        # decoder LSTM
        self.decoder = dy.CoupledLSTMBuilder(self.hyperparams['DEC_LAYERS'], self.hyperparams['INPUT_DIM_CHAR'] + self.dim, \
            self.dim, self.model)
        # decoder attention over chars
        self.W__char = self.model.add_parameters((self.dim, 2*self.dim_enc))

        self.att_func = self._att_head
        self.att_params_char = [self.W__char]

        # from char attention vector (2*dim_enc) and decoder state to updated decoder state
        self.W_h_char = self.model.add_parameters((self.dim, 2*self.dim_enc + self.dim))
        # softmax parameters
        self.R = self.model.add_parameters((self.hyperparams['VOCAB_SIZE_CHAR'], self.dim))
        self.bias = self.model.add_parameters(self.hyperparams['VOCAB_SIZE_CHAR'])

            
                
    def save_model(self, best_model_path):
        self.model.save(best_model_path)
 
    @staticmethod
    def _run_lstm(init_state, input_vecs):
        s = init_state
        out_vectors = []
        for vector in input_vecs:
          s = s.add_input(vector)
          out_vector = s.output()
          out_vectors.append(out_vector)
        return out_vectors

    @staticmethod
    def _att_head_unoptimized(query, input_vecs, att_mat):

        input_mat = dy.concatenate_cols(input_vecs) # dim*2 x seq_len
        input_mat_transf = att_mat * input_mat 
        unnormalized = dy.transpose(dy.transpose(query) * input_mat_transf)
        att_weights = dy.softmax(unnormalized)
        weighted_vector = input_mat*att_weights
        return weighted_vector, att_weights

    @staticmethod
    def _att_head(query, input_mat, input_mat_transf, *args):

        unnormalized = dy.transpose(dy.transpose(query) * input_mat_transf)
        att_weights = dy.softmax(unnormalized)
        weighted_vector = input_mat*att_weights
        return weighted_vector, att_weights

    def bilstm_transduce(self, fbuffRNN, bbuffRNN, embs):
        # returns a tuple:
        # the encoding for each char of the input sequence as a concat of the fwd and bwd LSTM vectors
        #  word encoding as a concatenation of the last fwd and last bwd LSTM vectors

        # BiLSTM forward pass
        char_fwds = self._run_lstm(fbuffRNN.initial_state(),embs)
        # BiLSTM backward pass
        char_bwds = self._run_lstm(bbuffRNN.initial_state(),reversed(embs))

        char_encoder = [dy.concatenate([fwd,bwd]) for fwd,bwd in zip(char_fwds,reversed(char_bwds))]
        word_encoder = dy.concatenate([char_fwds[-1],char_bwds[-1]])

        return char_encoder, word_encoder

    def param_init(self, instance, test_time=False): #initialize parameters for current cg with the current input
        
        #self.char_enoder = []
        self.char_encoder_mat = [] # caching
        self.char_encoder_mat_transf = [] # caching
        for item in instance:
            input2ind = item.input2ind
            #input_word,_,_,_,_,_,_= instance

            # biLSTM encoder of input string
            #word_padded = [BEGIN_CHAR] + [c for c in input_word] + [STOP_CHAR]
            #word2ind = [self.char_vocab.w2i.get(c, self.UNK) for c in word_padded]
            input_emb = [self.CHAR_VOCAB_LOOKUP[char_id] for char_id in input2ind]
            char_encoding, _ = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
            #self.char_enoder.append(char_encoding)
            # word representation as list of char biLSTM vectors
            #self.biencoder = char_encoding
            char_encoder_mat = dy.concatenate_cols(char_encoding) # dim*2 x seq_len
            #if not test_time:
            #    char_encoder_mat = dy.dropout(char_encoder_mat, self.hyperparams['DROPOUT'])
            self.char_encoder_mat.append(char_encoder_mat)

            char_encoder_mat_transf = self.W__char * char_encoder_mat
            self.char_encoder_mat_transf.append(char_encoder_mat_transf)

    def reset_decoder(self):
        self.s = self.decoder.initial_state()

        self.s = self.s.add_input(dy.concatenate([self.CHAR_VOCAB_LOOKUP[self.BEGIN], dy.ones((self.dim))]))


    def predict_next_(self, state, pos, scores=False, test_time=False, *args, **kwargs):
        # pos is position of the word in sentence
        query = state.output()

        #if not test_time:
        #    query = dy.dropout(query, self.hyperparams['DROPOUT'])
        # char context vector
        char_vector, char_att_weights = self.att_func(query, self.char_encoder_mat[pos], \
            self.char_encoder_mat_transf[pos], self.att_params_char)
        # updated decoder state
        h_output = dy.tanh(self.W_h_char * dy.concatenate([query, char_vector]))

        if not scores:
            return char_att_weights, dy.softmax(self.R * h_output + self.bias), h_output
        else:
            return char_att_weights, -dy.log_softmax(self.R * h_output + self.bias), h_output
    
    def consume_next_(self, state, pred_id, dec_state_updated):
        # decoder with input feeding
        new_state = state.add_input(dy.concatenate([self.CHAR_VOCAB_LOOKUP[pred_id], dec_state_updated]))
        return new_state
            
    def ll_loss(self, instance, *args):

        self.param_init(instance)
        instance_loss = []
        for pos,item in enumerate(instance):

            self.reset_decoder()
            #true_output2id = [self.char_vocab.w2i[a] for a in true_output]
            #true_output2id += [self.STOP]
            true_output2ind = item.output2ind
            losses = []
            for pred_id in true_output2ind:
                _, probs, dec_state_updated = self.predict_next_(self.s, pos)
                losses.append(-dy.log(dy.pick(probs, pred_id)))
                self.s = self.consume_next_(self.s, pred_id, dec_state_updated)
            instance_loss.append(dy.average(losses))
        return dy.average(instance_loss)

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        """Find k smallest elements of a matrix.
            Parameters
            ----------
            matrix : :class:`np.ndarray`
            The matrix.
            k : int
            The number of smallest elements required.
            Returns
            -------
            Tuple of ((row numbers, column numbers), values).
            """
        #flatten = matrix.flatten()
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]

    def predict(self, input, pos, beam_size):
        """Performs beam search.
            Parameters
            ----------
            input: str
            Input word
            pos: int
            Position of word in its sentence

            Returns
            -------
            outputs : list of lists of ints
            A list of the `beam_size` best sequences found in the order
            of decreasing likelihood.
            costs : list of floats
            A list of the costs for the `outputs`, where cost is the
            negative log-likelihood.
            """

        max_pred_len = max(MAX_PRED_SEQ_LEN,len(input)*3)
        self.reset_decoder()
        states = [self.s] * beam_size
        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = np.full(shape=(1,beam_size),fill_value=self.BEGIN,dtype = int)
        all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
        all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions
        all_pointers = np.zeros_like(all_outputs, dtype=int) # the positions of charachter attention pointers (for UNK replacement)
        
        prev_outputs = [self.BEGIN] * beam_size

        for i in range(max_pred_len):
            if all_masks[-1].sum() == 0:
                break
        
            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.

            #logprobs = np.array([-dy.log_softmax(self.predict_next_(s, position, scores=True)[1]).npvalue() for s in states])

            if beam_size>1:
                _pointers, _scores, _dec_out = zip(*[self.predict_next_(s, pos, scores=True, test_time=True) for s in states ])
                logprobs = np.array([s.npvalue() for s in _scores])
                pointers = [np.argmax(p.npvalue()) for p in _pointers]
            else:
                _pointers, _scores, _dec_out = self.predict_next_(states[0], pos, scores=True, test_time=True)
                logprobs = np.array(_scores.npvalue())
                pointers=[np.argmax(_pointers.npvalue())]


            next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :self.STOP] = np.inf
            next_costs[finished, self.STOP + 1:] = np.inf
            
            # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
            (indexes, outputs), chosen_costs = self._smallest(next_costs, beam_size, only_first_row=i == 0)
            # Rearrange everything
            new_states = (states[ind] for ind in indexes)
            #import pdb; pdb.set_trace()
            new_dec_outs = (_dec_out[ind] for ind in indexes)
            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]
            all_pointers = all_pointers[:, indexes]
            
            # Record chosen output and compute new states
            states = [self.consume_next_(s,pred_id,dec_out) for s, pred_id, dec_out in zip(new_states, outputs, new_dec_outs)]
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != self.STOP
            all_masks = np.vstack([all_masks, mask[None, :]])
            all_pointers = np.vstack([all_pointers, np.array(pointers)[indexes]])

        all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
        all_masks = all_masks[1:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
        all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
        all_pointers = all_pointers[1:-1]
        result = all_outputs, all_masks, all_costs, all_pointers

        return self.result_to_lists(self.char_vocab,result, input)
    
    @staticmethod
    def result_to_lists(vocab, result, input):
        UNK = vocab.w2i[UNK_CHAR] 
        outputs, masks, costs, pointers = [array.T for array in result]
        outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
        #words = [u''.join([vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
        words = []
        for output,pointer in zip(outputs,pointers):
            word=[]
            unk_replaces=False
            unk_pos=[]
            for i,pred_id in enumerate(output):
                if pred_id!=UNK:
                    word.append(vocab.i2w[pred_id])
                else:
                    unk_replaces=True
                    unk_pos.append(i)
                    word.append(input[pointer[i+1]]) # i+1 because of BOW padding
            pred_word = ''.join(word)
            if unk_replaces: print('Replaced unk at pos {} in prediction {} for input {}'.format(unk_pos,pred_word,input))
            words.append(pred_word)
        costs = list(costs.T.sum(axis=0))
        results = list(zip(costs, words))
        results.sort(key=lambda h: h[0])
        return results


class SoftAttentionWord(SoftAttention):
    def __init__(self, model_hyperparams, best_model_path=None):
        self.add_params(model_hyperparams)
        if best_model_path:
            self.model.populate(best_model_path)

    def add_params(self, model_hyperparams):
            super(SoftAttentionWord, self).add_params(model_hyperparams)

            self.dim_context = self.hyperparams['HIDDEN_DIM_CONTEXT']
            # BiLSTM for subword context 
            self.fbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_CONT_LAYERS'], 2*self.dim_enc, self.dim_context, self.model)
            self.bbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_CONT_LAYERS'], 2*self.dim_enc, self.dim_context, self.model)

            
            self.att_func = self._att_head
            self.att_params_char = [self.W__char]

            # from context vector (2*self.dim_context), char_vector and decoder state to updated decoder state
            self.W_h = self.model.add_parameters((self.dim, 2*self.dim_context + 2*self.dim_enc + self.dim))


    def param_init(self, instance, test_time=False): #initialize parameters for current cg with the current input
        
        self.char_encoder_mat = [] # caching
        self.char_encoder_mat_transf = [] # caching
        context_emb = []
        for item in instance:
            input2ind = item.input2ind
            input_emb = [self.CHAR_VOCAB_LOOKUP[char_id] for char_id in input2ind]
            char_encoding, word_encoding = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
            context_emb.append(word_encoding)

            char_encoder_mat = dy.concatenate_cols(char_encoding) # dim*2 x seq_len
            #if not test_time:
            #    char_encoder_mat = dy.dropout(char_encoder_mat, self.hyperparams['DROPOUT'])
            self.char_encoder_mat.append(char_encoder_mat)

            char_encoder_mat_transf = self.W__char * char_encoder_mat
            self.char_encoder_mat_transf.append(char_encoder_mat_transf)

        self.context_encoder, _ = self.bilstm_transduce(self.fbuffRNN_cont, self.bbuffRNN_cont, context_emb)

    def predict_next_(self, state, pos, scores=False, test_time=False, *args, **kwargs):

        query = state.output()
        if not test_time:
            query = dy.dropout(query, self.hyperparams['DROPOUT'])
        # char context vector
        char_vector, char_att_weights = self.att_func(query, self.char_encoder_mat[pos], \
            self.char_encoder_mat_transf[pos], self.att_params_char)

        # update state with char vector and context vector
        h_output = dy.tanh(self.W_h * dy.concatenate([query, self.context_encoder[pos], char_vector]))

        if not scores:
            return char_att_weights, dy.softmax(self.R * h_output + self.bias), h_output
        else:
            return char_att_weights, -dy.log_softmax(self.R * h_output + self.bias), h_output

class SoftAttentionWordGated(SoftAttention):
    def __init__(self, model_hyperparams, best_model_path=None):
        self.add_params(model_hyperparams)
        if best_model_path:
            self.model.populate(best_model_path)

    def add_params(self, model_hyperparams):
            super(SoftAttentionWordGated, self).add_params(model_hyperparams)

            self.dim_context = self.hyperparams['HIDDEN_DIM_CONTEXT']
            # BiLSTM for subword context 
            self.fbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_CONT_LAYERS'], 2*self.dim_enc, self.dim_context, self.model)
            self.bbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['ENC_CONT_LAYERS'], 2*self.dim_enc, self.dim_context, self.model)
            # self-attention over subwords
            #self.W__sub_self = self.model.add_parameters((2*self.dim_context, 2*self.dim_context))
            # decoder attention over subwords  
            self.W__cont = self.model.add_parameters((self.dim, 2*self.dim_context))

            self.att_func = self._att_head
            self.att_params_char = [self.W__char]
            self.att_params_cont = [self.W__cont]

            # from subword attention vector (2*self.dim_context) and decoder state to updated decoder state
            self.W_h_cont = self.model.add_parameters((self.dim, 2*self.dim_context + self.dim))
            # gate linear layer 
            self.W_gate = self.model.add_parameters((2, 2*self.dim_enc + self.dim + 2*self.dim_context))

    def param_init(self, instance, test_time=False): #initialize parameters for current cg with the current input
        

        self.char_encoder_mat = [] # caching
        self.char_encoder_mat_transf = [] # caching
        self.context_encoder_mat = [] # caching
        self.context_encoder_mat_transf = [] # caching


        context_emb = []
        for item in instance:
            input2ind = item.input2ind
            input_emb = [self.CHAR_VOCAB_LOOKUP[char_id] for char_id in input2ind]
            char_encoding, word_encoding = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
            context_emb.append(word_encoding)

            char_encoder_mat = dy.concatenate_cols(char_encoding) # dim*2 x seq_len
            #if not test_time:
            #    char_encoder_mat = dy.dropout(char_encoder_mat, self.hyperparams['DROPOUT'])
            self.char_encoder_mat.append(char_encoder_mat)

            char_encoder_mat_transf = self.W__char * char_encoder_mat
            self.char_encoder_mat_transf.append(char_encoder_mat_transf)

        context_encoder, _ = self.bilstm_transduce(self.fbuffRNN_cont, self.bbuffRNN_cont, context_emb)

        # for each word, it's context encoding is represented by other words in the sentence
        for pos,item in enumerate(instance):
            context_encoding = context_encoder[:pos]+context_encoder[pos+1:]
            context_encoder_mat = dy.concatenate_cols(context_encoding)
            self.context_encoder_mat.append(context_encoder_mat)

            context_encoder_mat_transf = self.W__cont * context_encoder_mat
            self.context_encoder_mat_transf.append(context_encoder_mat_transf)


    def predict_next_(self, state, pos, scores=False, test_time=False, *args, **kwargs):

        query = state.output()
        # if not test_time:
        #     query = dy.dropout(query, self.hyperparams['DROPOUT'])
        # char context vector
        char_vector, char_att_weights = self.att_func(query, self.char_encoder_mat[pos], \
            self.char_encoder_mat_transf[pos], self.att_params_char)
        # tag context vector
        context_vector,  context_att_weights = self.att_func(query, self.context_encoder_mat[pos], \
            self.context_encoder_mat_transf[pos], self.att_params_cont)
        # update state with char context vector and tag context vector
        h_cont = dy.tanh(self.W_h_cont * dy.concatenate([query, context_vector]))
        h_char = dy.tanh(self.W_h_char * dy.concatenate([query, char_vector]))

        gate_unnormalized = self.W_gate * dy.concatenate([query, context_vector, char_vector])
        gate = dy.softmax(gate_unnormalized)
        h_output = dy.concatenate([h_cont, h_char],d=1) *  gate

        #import pdb; pdb.set_trace()

        if not scores:
            return char_att_weights, dy.softmax(self.R * h_output + self.bias), h_output
        else:
            return char_att_weights, -dy.log_softmax(self.R * h_output + self.bias), h_output

class SoftAttentionSubw(SoftAttentionWordGated):
    def __init__(self, model_hyperparams, best_model_path=None):
        self.add_params(model_hyperparams)
        if best_model_path:
            self.model.populate(best_model_path)

    def add_params(self, model_hyperparams):
            super(SoftAttentionSubw, self).add_params(model_hyperparams)

    def param_init(self, instance, test_time=False): #initialize parameters for current cg with the current input
        

        self.char_encoder_mat = [] # caching
        self.char_encoder_mat_transf = [] # caching
        self.context_encoder_mat = [] # caching
        self.context_encoder_mat_transf = [] # caching
        subw_emb = []
        for item in instance:
            input2ind = item.input2ind
            bpe_len = item.subw_len

            input_emb = [self.CHAR_VOCAB_LOOKUP[char_id] for char_id in input2ind]
            char_encoding, word_encoding = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)

            char_encoder_mat = dy.concatenate_cols(char_encoding) # dim*2 x seq_len
            #if not test_time:
            #    char_encoder_mat = dy.dropout(char_encoder_mat, self.hyperparams['DROPOUT'])
            self.char_encoder_mat.append(char_encoder_mat)

            char_encoder_mat_transf = self.W__char * char_encoder_mat
            self.char_encoder_mat_transf.append(char_encoder_mat_transf)

            # word representation as subwords vectors (average c-biLSTM within segments)
            start_pos = np.cumsum([0]+bpe_len[:-1])
            end_pos = np.cumsum(bpe_len)
            
            subw_emb_ = [dy.average(char_encoding[start:end]) for start,end in zip(start_pos,end_pos)]
            subw_emb.extend(subw_emb_)

        subs_encoding, _ = self.bilstm_transduce(self.fbuffRNN_cont, self.bbuffRNN_cont, subw_emb)

        # for each word, it's context encoding is represented by other subwords in the sentence

        for item in instance:
            subw_pos_start, subw_pos_end = item.subw_pos
            context_encoding = subs_encoding[:subw_pos_start]+subs_encoding[subw_pos_end:]
            context_encoder_mat = dy.concatenate_cols(context_encoding)
            self.context_encoder_mat.append(context_encoder_mat)

            context_encoder_mat_transf = self.W__cont * context_encoder_mat
            self.context_encoder_mat_transf.append(context_encoder_mat_transf)


def evaluate(model, data, beam):
    # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    correct_tag = 0
    final_results = []
    data_len = 0
    for i, batch in enumerate(data):
        for instance in batch:
            dy.renew_cg(autobatching=False)
            model.param_init(instance, test_time=True)
            k=0
            for pos,item in enumerate(instance):
                k+=1
                input_word = item.input
                output = item.output
                predictions = model.predict(input_word, pos, beam) # list of (cost, prediction) sorted by cost
                prediction = predictions[0][1]
                if prediction == output:
                    correct += 1
                if i ==0 and k < 5: print(input_word, output, prediction)
                final_results.append((input_word,prediction))
                data_len += 1
    accuracy = correct / data_len
    accuracy_dict = {'Accuracy':accuracy}
    return accuracy_dict, final_results


def model_training(model_hyperparams, model, trainer, objective, epochs, train_data, dev_data, train_hyperparams, best_model_path, \
                    best_dev_accuracy=-1., debug=False):

    sanity_set_size = 50 # for speed - check prediction accuracy on train set
    patience = 0
    
    

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    
    header_info = ('epoch', 'avg_train_loss', 'train_acc', 'dev_acc')
    log_to_file(log_file_name, '\t'.join(str(x) for x in header_info))
    #trainer.status()

    done = False
    print('Start training...')
    #for epoch in range(epochs):
    while ((not done) and epoch < epochs):    

        # compute loss for each sample and update
        train_loss = 0.  # total train loss
        avg_train_loss = 0.  # avg training loss

        #import pdb; pdb.set_trace()
        
        if debug:
            train_data_epoch = train_data.iter(shuffle=True, indices=sanity_set_size, batch_size=train_hyperparams['BATCH_SIZE'])
            #dev_data_eval = dev_data.iter(indices=sanity_set_size)
            dev_data_eval = dev_data.iter()
        else:
            train_data_epoch = train_data.iter(shuffle=True, batch_size=train_hyperparams['BATCH_SIZE'])
            dev_data_eval = dev_data.iter()

        # new graph for each batch
        batch_loss = []
        #dy.renew_cg()
        report_freq = train_hyperparams['REPORT_FREQ']
        batches_number_total = len(train_data_epoch)
        print('Number of bathces: {}'.format(batches_number_total))

        batch_number = 0
        for i, batch in enumerate(train_data_epoch):
            then = time.time()
            i+=1
            dy.renew_cg(autobatching=True)
            for instance in batch:
                loss = objective(instance)
                batch_loss.append(loss)
            total_batch_loss = dy.esum(batch_loss)
            train_loss += total_batch_loss.scalar_value()
            total_batch_loss = total_batch_loss /len(batch_loss)
            total_batch_loss.backward()
            trainer.update()
            batch_loss = []
            #dy.renew_cg()
            if i == batches_number_total or  i%report_freq == 0:
                avg_train_loss = train_loss /train_data.length

                print('\t...finished epoch {} batches {} in {:.3f} sec'.format(epoch, i, time.time() - then))
                #trainer.status()

                # get train accuracy
                print('evaluating on train...')
                dy.renew_cg(autobatching=False) # new graph for all the examples
                then = time.time()
                train_accuracy_dict, _ = evaluate(model, train_data.iter(indices=sanity_set_size), \
                    beam=train_hyperparams['BEAM'])
                print('\t...finished in {:.3f} sec'.format(time.time() - then))

                # get dev accuracy
                print('evaluating on dev...')
                then = time.time()
                dy.renew_cg(autobatching=False) # new graph for all the examples
                dev_accuracy_dict, _ = evaluate(model, dev_data_eval, \
                    beam=train_hyperparams['BEAM'])
                print('\t...finished in {:.3f} sec'.format(time.time() - then))

                dev_accuracy = dev_accuracy_dict['Accuracy']
                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    # save best model
                    ti.save_model(best_model_path)
                    print('saved new best model to {}'.format(best_model_path))
                    patience = 0
                else:
                    patience += 1



                log_info = (':'.join([str(epoch),str(i)]), avg_train_loss, train_accuracy_dict['Accuracy'], dev_accuracy_dict['Accuracy'])
                print('epoch-batch: {} train loss: {:.4f} train accuracy: {:.4f} dev accuracy: {:.4f} \
                            best dev accuracy: {:.4f} patience = {}'.format(\
                            *log_info, best_dev_accuracy, patience))
                log_to_file(log_file_name, '\t'.join(str(x) for x in log_info))

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    done = True
                    break

                if patience == train_hyperparams['PATIENCE']:
                    print('out of patience after {} epochs'.format(epoch))
                    train_progress_bar.finish()
                    done = True
                    break
                    
        # finished epoch
        train_progress_bar.update(epoch)
        epoch += 1
                
    print('finished training.')
    return 


if __name__ == "__main__":
    arguments = docopt(__doc__)
    #print(arguments)
    
    np.random.seed(123)
    random.seed(123)
    
    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)
        

    data_set = DataSet
    
    if arguments['train']:
        
        print('=========TRAINING:=========')

        dataset_reader = extract_data
        if arguments['--subw_context']:
            model_class = SoftAttentionSubw
            dataset_reader = extract_data_with_subw
        elif arguments['--word_context']:
            model_class = SoftAttentionWord
        elif arguments['--word_context_gated']:
            model_class = SoftAttentionWordGated
        else:
            model_class = SoftAttention
        
        assert (arguments['--train_path']!=None) & (arguments['--dev_path']!=None)
        
        # load data
        print('Loading data...')
        
        train_path = check_path(arguments['--train_path'], 'train_path')
        dev_path = check_path(arguments['--dev_path'], 'dev_path')

        if arguments['--subw_context']:
            assert (arguments['--train_bpe_path']!=None) & (arguments['--dev_bpe_path']!=None)    
            train_bpe_path = check_path(arguments['--train_bpe_path'], 'train_bpe_path')
            dev_bpe_path = check_path(arguments['--dev_bpe_path'], 'dev_bpe_path')
        else:
            train_bpe_path, dev_bpe_path = None, None

        train_data = data_set.from_file(train_path, train_bpe_path, dataset_reader, lower_target=arguments['--lower_target'])
        print('Train data has {} examples'.format(train_data.length))

        dev_data = data_set.from_file(dev_path, dev_bpe_path, dataset_reader, lower_target=arguments['--lower_target'])
        print('Dev data has {} examples'.format(dev_data.length))
    
        for data in [train_data, dev_data]:
            words = data.inputs() + data.outputs()
            #print(len(words))
            c_data = set([c for w in words for c in w])
            #print(c_data)
            for c in SPECIAL_CHARS:
              if c in c_data: print('Char {} in data'.format(c))
              assert c not in c_data


        # Paths for checks and results
        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best'
        

        char_vocab_path = check_vocab_path(arguments['--char_vocab_path'], model_folder)
        char_data = set(train_data.inputs() + train_data.outputs())
        build_vocabulary(char_data, char_vocab_path, special_symbols_list=SPECIAL_CHARS, min_freq=int(arguments['--char_min_freq']))
        char_vocab = Vocab.from_file(char_vocab_path)
        train_data.encode(char_vocab)
        dev_data.encode(char_vocab)

        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM_CHAR': int(arguments['--char_input']),
                            'INPUT_DIM_TAG': int(arguments['--tag_input']),
                            'HIDDEN_DIM_ENC': int(arguments['--hidden_enc']),
                            'HIDDEN_DIM_DEC': int(arguments['--hidden_dec']),
                            'HIDDEN_DIM_CONTEXT': int(arguments['--hidden_context']),
                            'ENC_LAYERS': int(arguments['--enc_layers']),
                            'DEC_LAYERS': int(arguments['--dec_layers']),
                            'ENC_CONT_LAYERS': int(arguments['--enc_cont_layers']),
                            'DROPOUT': float(arguments['--dropout']),
                            'VOCAB_SIZE_CHAR': char_vocab.size(),
                            'VOCAB_PATH_CHAR': char_vocab_path,
                            'Subword_context': arguments['--subw_context'],
                            'Word_context_gated': arguments['--word_context_gated'],
                            'Word_context': arguments['--word_context'],
                            'char_min_freq': int(arguments['--char_min_freq'])}

        
        ti = model_class(model_hyperparams)

        objective = ti.ll_loss

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'BEAM': int(arguments['--beam']),
                            'BATCH_SIZE': int(arguments['--batch_size']),
                            'REPORT_FREQ': int(arguments['--report_freq']),
                            'TRAIN_PATH': train_path,
                            'DEV_PATH': dev_path}


        write_param_file(output_file_path, {**model_hyperparams,**train_hyperparams})
        
        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(ti.model)

        epochs = train_hyperparams['EPOCHS']
        model_training(model_hyperparams, ti, trainer, objective, epochs, \
            train_data, dev_data, train_hyperparams, \
              best_model_path, \
              debug = arguments['--verbose'])

        # final evaluation
        print('final evaluation of best model on dev set..')
        ti = model_class(model_hyperparams, best_model_path)
        dev_accuracy, dev_results = evaluate(ti, dev_data.iter(), beam=train_hyperparams['BEAM'])
        write_pred_file(output_file_path, dev_results)
        write_eval_file(output_file_path, dev_accuracy, dev_path)
        print('Best dev accuracy: {}'.format(dev_accuracy))
        
    elif arguments['test']:
        print('=========EVALUATION ONLY:=========')
        # requires test path, model path of pretrained path and results path where to write the results to

        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = os.path.join(model_folder,arguments['--pred_path'])
        hypoparams_file = model_folder + '/best'
        
        hypoparams_file_reader = open(hypoparams_file, 'r')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])

        model_hyperparams = {'INPUT_DIM_CHAR': int(hyperparams_dict['INPUT_DIM_CHAR']),
                            'INPUT_DIM_TAG': int(hyperparams_dict['INPUT_DIM_TAG']),
                            'HIDDEN_DIM_ENC': int(hyperparams_dict['HIDDEN_DIM_ENC']),
                            'HIDDEN_DIM_DEC': int(hyperparams_dict['HIDDEN_DIM_DEC']),
                            'HIDDEN_DIM_CONTEXT': int(hyperparams_dict['HIDDEN_DIM_CONTEXT']),
                            'ENC_LAYERS': int(hyperparams_dict['ENC_LAYERS']),
                            'DEC_LAYERS': int(hyperparams_dict['DEC_LAYERS']),
                            'ENC_CONT_LAYERS': int(hyperparams_dict['ENC_CONT_LAYERS']),
                            'DROPOUT': float(hyperparams_dict['DROPOUT']),
                            'VOCAB_PATH_CHAR': hyperparams_dict['VOCAB_PATH_CHAR'],
                            'VOCAB_SIZE_CHAR': INT(hyperparams_dict['VOCAB_SIZE_CHAR']),
                            'Subword_context': True if hyperparams_dict['Subword_context']=="True" else False,
                            'Word_context_gated': True if hyperparams_dict.get('Word_context_gated','False')=="True" else False,
                            'Word_context': True if hyperparams_dict['Word_context']=="True" else False}
        # a fix for vocab path when transferring files b/n vm
        model_hyperparams['VOCAB_PATH_CHAR'] = check_path(model_folder + '/char_vocab.txt', 'vocab_path', is_data_path=False)


        dataset_reader = extract_data
        if model_hyperparams['Subword_context']:
            model_class = SoftAttentionSubw
            dataset_reader = extract_data_with_subw
        elif model_hyperparams['Word_context']:
            model_class = SoftAttentionWord
        elif model_hyperparams['Word_context_gated']:
            model_class = SoftAttentionWordGated
        else:
            model_class = SoftAttention

        print('Loading data...')
        assert arguments['--test_path']!=None
        test_path = check_path(arguments['--test_path'], '--test_path')

        if arguments['--subw_context']:
            assert arguments['--test_bpe_path']!=None
            test_bpe_path = check_path(arguments['--test_bpe_path'], '--test_bpe_path')
        else:
            test_bpe_path = None
        
        test_data = data_set.from_file(test_path, test_bpe_path, dataset_reader, lower_target=arguments['--lower_target'])
        print('Test data has {} examples'.format(test_data.length))

        char_vocab = Vocab.from_file(model_hyperparams['VOCAB_PATH_CHAR'])
        test_data.encode(char_vocab)
        

        words = test_data.inputs() + test_data.outputs()
        c_data = set([c for w in words for c in w])
        for c in SPECIAL_CHARS:
            if c in c_data: print('Char {} in data'.format(c))
            assert c not in c_data


        ti = model_class(model_hyperparams, best_model_path)

        print('Evaluating on test..')
        t = time.clock()
        accuracy, test_results = evaluate(ti, test_data.iter(), int(arguments['--beam']))

        print('Time: {}'.format(time.clock()-t))
        print('accuracy: {}'.format(accuracy))
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)
