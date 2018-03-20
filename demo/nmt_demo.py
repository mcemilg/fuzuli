#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np
from lstm_chars import poem_data as p_data

'''
    File: nmt_demo.py
    Author: M.Cemil Guney
    Description: The demo of nmt model. Finds 'vezin' style of given lines.
'''
if (len(sys.argv) < 2):
    print("Usage: python nmt.py data_folder")
    sys.exit()

# read poems
vezins, lines = p_data.read_all_poems(sys.argv[1])

# create dictionaries
lines_dict, lines_reverse_dict = p_data.build_dictionary(lines, vezins, key="lines")
vezins_dict, vezins_reverse_dict = p_data.build_dictionary(lines, vezins, key="vezins")

# add special keys
start_token_str = "<go>"
end_token_str = "</go>"
vezins_dict, vezins_reverse_dict = p_data.add_key_to_dict(vezins_dict, vezins_reverse_dict, start_token_str, key="max")
vezins_dict, vezins_reverse_dict = p_data.add_key_to_dict(vezins_dict, vezins_reverse_dict, end_token_str,  key="max")

# create datas 
encoder_inp_data, decoder_inp_data, target_data = p_data.nmt_create_data(lines, vezins)


# convert data number format
enc_inp_num_data = p_data.convert_to_num(encoder_inp_data, lines_dict)
dec_inp_num_data = p_data.convert_to_num(decoder_inp_data, vezins_dict)
target_num_data = p_data.convert_to_num(target_data, vezins_dict)

#parameters
inp_seq_length = len(enc_inp_num_data[0])
dec_inp_seq_length = len(dec_inp_num_data[0])
inp_alphabet_size = len(lines_dict)
out_alphabet_size = len(vezins_dict)
max_length = p_data.get_max_length(lines, "lines")
batch_size = 20

# demo checkpoint
check_point = "checkpoints/nmt_v2_train_1521545110-0"


def get_input(iterator):

    content = input("Mısra : ")
    
    content = [x for x in content if x != []]

    content = p_data.padding_vector(content, max_length, padding=" ")

    content_batch = np.full((batch_size-1, max_length), [' '], dtype=list)
    content_batch = content_batch.tolist()
    content_batch = [content] + content_batch

    num_batch = p_data.convert_to_num(content_batch, lines_dict)

    encoder_inp = iterator.transpose(num_batch)
    seqlen = iterator.get_seqlen(num_batch)

    return encoder_inp, seqlen

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(check_point + '.meta')
    new_saver.restore(sess, check_point)

    # graph tensosrs
    scope = "root_1/"
    infer_prediction = scope + "prediction:0"
    infer_outputs = scope + "Shape:0"
    infer_encoder_inputs = scope + "inputs:0"
    infer_seqlen = scope + "seqlen:0"

    iterator = p_data.NmtBatchIterator(enc_inp_num_data, dec_inp_num_data, target_num_data, batch_size)


    while True:

        encoder_inp, seq_length = get_input(iterator) 

        #encoder_inp, decoder_inp, trgt, seq_length, trgt_seq_length, t= iterator.next()
 
        predictions, _ = sess.run([infer_prediction, infer_outputs], feed_dict={
                   infer_encoder_inputs:list(encoder_inp),
                   infer_seqlen:list(seq_length)})

        for p in predictions:
            pred = ""
            for i in p:
                pred += vezins_reverse_dict[i]
            
            print(pred)
            break
    
        
        #p_data.print_result(pred, t, vezins_reverse_dict)



