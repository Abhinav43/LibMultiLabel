#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log.out 2>&1


for domain in tf_roberta-large-nli-stsb-mean-tokens_.embed tf_bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16_.embed tf_xlm-roberta-large_.embed tf_bert-large-uncased_.embed tf_bert-base-nli-mean-tokens.embed tf_bioelectra-base-discriminator-pubmed_.embed tf_scibert_scivocab_uncased_.embed tf_bert-large-nli-mean-tokens.embed tf_bert-base-uncased.embed tf_BiomedNLP-PubMedBERT-base-uncased-abstract_.embed tf_xlm-roberta-base_.embed vocab.csv tf_biobert_v1.0_pubmed_pmc_.embed tf_paraphrase-xlm-r-multilingual-v1_.embed tf_roberta-base_.embed Untitled.ipynb tf_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_.embed tf_roberta-large_.embed tf_roberta-large-mnli_.embed tf_bluebert_pubmed_mimic_uncased_L-12_H-768_A-12_.embed .ipynb_checkpoints tf_bio_roberta-base_pubmed_.embed tf_albert-base-v2.embed tf_sentence_bert_.embed tf_bigbird-base-mimic-mortality_.embed tf_paraphrase-multilingual-mpnet-base-v2_.embed tf_LaBSE.embed tf_bluebert_pubmed_uncased_L-12_H-768_A-12_.embed
do
    python3 main.py --config example_config/MIMIC-50/bigru.yml --train_path data/MIMIC-50/train.txt --test_path data/MIMIC-50/test.txt --val_path data/MIMIC-50/test.txt --embed_file /home/admin/Monk/gene_emd/tf_embds/$domain
done


