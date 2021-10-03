#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log_4.out 2>&1


for domain in amax amin logsum mean norm pod std sum last covd1d both

do
  a1=( use_m_4.pkl use_l_4.pkl use_l_5.pkl use_m_word_2.pkl use_m_word_4.pkl use_m_word_3.pkl use_l_word_4.pkl use_l_word_5.pkl )
  a2=( processed_full.embed processed_full.embed processed_full.embed processed_full.embed processed_full.embed processed_full.embed processed_full.embed processed_full.embed )

  declare -i i=0

  while [ "${a1[i]}" -a "${a2[i]}" ]; do

      python3 main.py --config example_config/MIMIC-50/bigru.yml --train_path data/MIMIC-50/train.txt --test_path data/MIMIC-50/test.txt --val_path data/MIMIC-50/test.txt --embed_file all_emb_data/s_emb/${a2[i]} --gcn_file all_emb_data/gcn_data_4/${a1[i]} --model_attach_mode ${domain} --gcn_dim 1 --gpu_id 3
      ((i++))

  done
done
