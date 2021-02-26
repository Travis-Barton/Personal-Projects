#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:54:34 2020

@author: tbarton
"""
import sys
import subprocess
import os
# os.chdir('machine_learning_scripts/speed_test')
import tensorflow as tf

if tf.__version__ == '2.0.0':
    print('wrong tf version')
    subprocess.check_call([sys.executable, 'conda', 'install', 'tensorflow==1.15.0'])
import gpt_2_simple as gpt2

gpt2.download_gpt2(model_name='774m')
# gpt2.download_gpt2()


sess = gpt2.start_tf_sess()
#sess = gpt2.load_gpt2(sess, 'second_run', multi_gpu=True)
gpt2.finetune(sess,
              dataset='big_chess_set.txt',
              run_name='new_run_large',
              print_every=1,
              multi_gpu=True,
              save_every=100,
              combine=100,
              steps=10000)   # steps is max number of training steps

# print('readying to generate!')
# single_text = gpt2.generate(sess, prefix='e4, e5 ', return_as_list=True, run_name='large_run')[0]

print('readying to generate!')
single_text = gpt2.generate(sess, prefix='e4, e5 ', return_as_list=True, run_name='second_run')[0]
sess.close()
print(single_text)      






