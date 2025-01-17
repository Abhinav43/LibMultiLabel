#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log_2.out 2>&1


for domain in amax amin logsum mean norm pod std sum last covd1d both

do
  a1=( glove_word_50.pkl glove_sentence_50.pkl glove_word_100.pkl glove_sentence_100.pkl glove_word_300.pkl glove_sentence_300.pkl w2v_word_50.pkl w2v_word_100.pkl w2v_word_300.pkl w2v_sentence_50.pkl )
  a2=( glove_50_None.embed glove_50_None.embed glove_100_None.embed glove_100_None.embed glove_300_None.embed glove_300_None.embed processed_full.embed processed_full.embed processed_full.embed processed_full.embed )

  declare -i i=0

  while [ "${a1[i]}" -a "${a2[i]}" ]; do

      python3 main.py --config example_config/MIMIC-50/bigru.yml --train_path data/MIMIC-50/train.txt --test_path data/MIMIC-50/test.txt --val_path data/MIMIC-50/test.txt --embed_file all_emb_data/s_emb/${a2[i]} --gcn_file all_emb_data/gcn_data_4/${a1[i]} --model_attach_mode ${domain} --gcn_dim 1 --gpu_id 1 2>&1 | tee ./output_results/${a2[i]}_${a1[i]}_${domain}.txt
      ((i++))

  done
done
