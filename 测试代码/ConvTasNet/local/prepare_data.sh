#!/bin/bash

storage_dir=/run/media/oneran/windows
n_src=2
python_path=/bin/python

/home/oneran/Wave-Split-CSU2021/测试代码/ConvTasNet/utils/parse_options.sh

$python_path /home/oneran/Wave-Split-CSU2021/测试代码/ConvTasNet/local/create_local_metadata.py --librimix_dir $storage_dir/Libri2Mix

$python_path /home/oneran/Wave-Split-CSU2021/测试代码/ConvTasNet/local/get_text.py \
  --libridir $storage_dir/LibriSpeech \
  --split test-clean \
  --outfile /home/oneran/Wave-Split-CSU2021/测试代码/ConvTasNet/data/test_annotations.csv
