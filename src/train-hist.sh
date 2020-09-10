# Train neural model with context on a histirical dataset
#./train-hist-batch-word.sh no-context
#./train-hist-batch-word.sh word-context
#./train-hist-batch-word.sh word-context-gated
#./train-hist-batch-word.sh subword-context
# add second argument 'print' to print commands for bpe preparation, when using subword-context

# RESULTS
export lang_res_folder=/gennorm/hist-lemmatization/lcct/context/

mkdir -p lang_res_folder

# DATA 
export lang_data_folder=/home/ubuntu/pie-data/historical/LLCT1/

export TRAIN=${lang_data_folder}train.tsv
export DEV=${lang_data_folder}dev.tsv
export TEST=${lang_data_folder}test.tsv

export MODEL='context_subw'
export BATCH_SIZE=1


#head $lang_data_folder/dev.bpe

####### Train neural model

export SEED=1
export MODEL_FOLDER=${lang_res_folder}/$1
export MODEL_PATH=${MODEL_FOLDER}_b${BATCH_SIZE} # will be diffferent for ensemble
export RESULTS=${MODEL_FOLDER}_${BATCH_SIZE} # will be diffferent for ensemble
export BEAM=3

if [[ $1 == 'no-context' ]]; then
echo python ${MODEL}.py train --dynet-mem 5000 --dynet-seed $SEED --dynet-autobatch 1 \
--train_path=$TRAIN --dev_path=$DEV  ${MODEL_PATH} \
  --beam=$BEAM --batch_size=1  --char_input=300 --hidden_enc=300 --hidden_dec=600
elif [[ $1 == 'word-context' ]]; then 
echo python ${MODEL}.py train --dynet-mem 5000 --dynet-seed $SEED --dynet-autobatch 1 \
--train_path=$TRAIN --dev_path=$DEV  ${MODEL_PATH} \
  --beam=$BEAM --batch_size=1  --char_input=300 --hidden_enc=300 --hidden_dec=600 --word_context --hidden_context=150
elif [[ $1 == 'word-context-gated' ]]; then 
echo python ${MODEL}.py train --dynet-mem 5000 --dynet-seed $SEED --dynet-autobatch 1 \
--train_path=$TRAIN --dev_path=$DEV  ${MODEL_PATH} \
  --beam=$BEAM --batch_size=1  --char_input=300 --hidden_enc=300 --hidden_dec=600 --word_context_gated --hidden_context=150
elif [[ $1 == 'subword-context' ]]; then 

# ###### Segment source side with bpe

# extract source side tokens
export TRAIN_SRC=$lang_res_folder/train.src
export DEV_SRC=$lang_res_folder/dev.src
export TEST_SRC=$lang_res_folder/test.src

if [[ $2 == 'print' ]]; then
echo python tools/preprocess.py extract_src $DEV $DEV_SRC
echo python tools/preprocess.py extract_src $TRAIN $TRAIN_SRC
echo python tools/preprocess.py extract_src $TEST $TEST_SRC

else
python tools/preprocess.py extract_src $DEV $DEV_SRC
python tools/preprocess.py extract_src $TRAIN $TRAIN_SRC
python tools/preprocess.py extract_src $TEST $TEST_SRC
fi

# # train bpe with wiki data
export bpe_n=3000
#export bpe_wiki_data=~/ha-seg/seg-to-seg/data/wiki2018/latvian/word-list.txt
export bpe_model=$lang_res_folder/bpe-codes-$bpe_n.src

if [[ $2 == 'print' ]]; then
echo python tools/learn_bpe.py -s $bpe_n <  $TRAIN_SRC > $bpe_model
else
python tools/learn_bpe.py -s $bpe_n <  $TRAIN_SRC > $bpe_model
fi

# # apply bpe to the extracted source side tokens
export TRAIN_BPE=${lang_data_folder}train-bpe-${bpe_n}
export DEV_BPE=${lang_data_folder}dev-bpe-${bpe_n}
export TEST_BPE=${lang_data_folder}test-bpe-${bpe_n}

if [[ $2 == 'print' ]]; then
echo python tools/apply_bpe.py -c $bpe_model < $DEV_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $DEV_BPE
echo python tools/apply_bpe.py -c $bpe_model < $TRAIN_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $TRAIN_BPE
echo python tools/apply_bpe.py -c $bpe_model < $TEST_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $TEST_BPE
else
python tools/apply_bpe.py -c $bpe_model < $DEV_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $DEV_BPE
python tools/apply_bpe.py -c $bpe_model < $TRAIN_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $TRAIN_BPE
python tools/apply_bpe.py -c $bpe_model < $TEST_SRC |sed -r 's/(@@ )|(@@ ?$)/|/g' > $TEST_BPE
fi


echo python ${MODEL}.py train --dynet-mem 5000 --dynet-seed $SEED --dynet-autobatch 1 \
--train_path=$TRAIN --dev_path=$DEV --train_bpe_path=$TRAIN_BPE --dev_bpe_path=$DEV_BPE ${MODEL_PATH} \
  --beam=$BEAM --batch_size=1  --char_input=300 --hidden_enc=300 --hidden_dec=600 --subw_context --hidden_context=150

else
echo "Unknown configuration!"

fi

####### Evaluate neural model
# run eval
echo python ${MODEL}.py test ${MODEL_PATH} --test_path=$DEV --test_bpe_path=$DEV_BPE --beam=$BEAM \
--pred_path=best.dev.$BEAM 
echo python ${MODEL}.py test ${MODEL_PATH} --test_path=$TEST --test_bpe_path=$TEST_BPE --beam=$BEAM \
--pred_path=best.test.$BEAM

