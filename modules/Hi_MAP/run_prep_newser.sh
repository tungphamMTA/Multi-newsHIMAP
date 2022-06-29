python preprocess.py -train_src ./../../preprocessed_truncated/train.txt.src.tokenized.fixed.cleaned.final.truncated.txt \
                     -train_tgt ./../../preprocessed_truncated/train.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt \
                     -valid_src ./../../preprocessed_truncated/val.txt.src.tokenized.fixed.cleaned.final.truncated.txt \
                     -valid_tgt ./../../preprocessed_truncated/val.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt \
                     -save_data newser_sent_500/newser_sents \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 300 \
                     -dynamic_dict \
                     -share_vocab \
                     -max_shard_size 10000000