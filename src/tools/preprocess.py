#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Preprocessing procedures: 
	Extracts tokens from input side of conll data (extract_src)
	Masks capitalization chars (mask_cap)

Usage:
  preprocess.py extract_src
  FILE_IN FILE_OUT

Arguments:
  FILE_IN     path to input file
  FILE_OUT    path to output file

"""

from docopt import docopt
import sys


def extract_data_with_subw(filename, bpe_filename, *args, **kwargs):
    """ Load data from conll file
        """    

        
    input_col, output_col = 0,1

    MAX_WORD_COUNT_PER_SENT=35
    
    f = open(filename)
    f_bpe = open(bpe_filename)
    dataset = []
    no_cont_inst_count = 0
    subw_pos_start = 0
    inputs, outputs, subw_lens, subw_positions = [], [], [], []
    # subw_lens (list of list of int) keeps info about length of subwords per word \
    # subw_positions (list of list of int) - subword posistions of each word in a flat subword list
    word_per_sent_count = 0 
    for i,(line, line_bpe) in enumerate(zip(f,f_bpe)):
        if i>0: #skip header
            if len(line.strip())==0 or word_per_sent_count == MAX_WORD_COUNT_PER_SENT:
              #if word_per_sent_count == MAX_WORD_COUNT_PER_SENT:
                # update dataset
                  tup = (inputs, outputs, subw_lens, subw_positions)
                  if word_per_sent_count>1:
                    dataset.append(tup)
                  else:
                    no_cont_inst_count+=1
                  inputs, outputs, subw_lens, subw_positions = [], [], [], []
                  subw_pos_start = 0
                  word_per_sent_count = 0
            if len(line.strip())!=0:
                # collect tokens in one sentence
                  splt = line.strip().split('\t')
                  input_string = splt[input_col]
                  output_string = splt[output_col]
                  inputs.append(input_string)
                  outputs.append(output_string)
                  # length of segments
                  subw_src = line_bpe.strip().split('|')
                  subw_len = [len(split) for split in subw_src]
                  subw_lens.append(subw_len)
                  subw_pos = (subw_pos_start,subw_pos_start+len(subw_len))
                  subw_positions.append(subw_pos)
                  subw_pos_start += len(subw_len)
                  word_per_sent_count += 1 
                
    if word_per_sent_count > 0:
        tup = (inputs, outputs, subw_lens, subw_positions)
        if word_per_sent_count>1:
          dataset.append(tup)
        else:
            no_cont_inst_count+=1
    print('found', len(dataset), 'instances')
    print('removed {} instance with no context'.format(no_cont_inst_count))
    return dataset

def extract_data(filename, *args, **kwargs):
    """ Load data from conll file
        """    

    #print('use_morf_tags: ',use_morf_tags)
        
    input_col, output_col = 0,1

    MAX_WORD_COUNT_PER_SENT=35
    
    f = open(filename)
    dataset = []
    no_cont_inst_count = 0
    inputs, outputs = [], []
    word_per_sent_count = 0 
    for i,(line) in enumerate(f):
        if i>0: #skip header
            if len(line.strip())==0 or word_per_sent_count == MAX_WORD_COUNT_PER_SENT:
              #if word_per_sent_count == MAX_WORD_COUNT_PER_SENT:
                # update dataset
                  tup = (inputs, outputs)
                  if word_per_sent_count>1:
                    dataset.append(tup)
                  else:
                    no_cont_inst_count+=1
                  inputs, outputs = [], []
                  word_per_sent_count = 0
            if len(line.strip())!=0:
                # collect tokens in one sentence
                  splt = line.strip().split('\t')
                  input_string = splt[input_col]
                  output_string = splt[output_col]
                  inputs.append(input_string)
                  outputs.append(output_string)
                  word_per_sent_count+=1
                
    if word_per_sent_count > 0:
        tup = (inputs, outputs)
        if word_per_sent_count>1:
          dataset.append(tup)
        else:
            no_cont_inst_count+=1
    print('found', len(dataset), 'instances')
    print('removed {} instance with no context'.format(no_cont_inst_count))
    return dataset

def extract_src(file_in, file_out):
  '''Extract source side and prepares input for bpe segmentation
  '''
  input_col, output_col = 0,1

  for line in file_in:
    if len(line.strip())==0:
      file_out.write('\n')
    else:
      splt = line.strip().split('\t')
      file_out.write(splt[input_col]+'\n')
  return

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    
    assert arguments['FILE_IN']!=None
    assert arguments['FILE_OUT']!=None

    file_in = open(arguments['FILE_IN'],'r')
    file_out = open(arguments['FILE_OUT'],'w')


    if arguments['extract_src']:
      extract_src(file_in, file_out)

