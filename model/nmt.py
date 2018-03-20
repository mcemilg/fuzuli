import sys
import poem_data as p_data
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
import math
import time
from random import shuffle



'''
    File: nmt.py
    Author: M.Cemil Guney
    Description: Basic implementation of encoder-decoder model for translating styles of old Turkish poems.
'''


if (len(sys.argv) < 2):
    print("Usage: python nmt.py data_folder")
    sys.exit()


print("Reading data...")

# read poems
vezins, lines = p_data.read_all_poems(sys.argv[1])

print("Prepearing data...")

# create dictionaries
lines_dict, lines_reverse_dict = p_data.build_dictionary(lines, vezins, key="lines")
vezins_dict, vezins_reverse_dict = p_data.build_dictionary(lines, vezins, key="vezins")
#print( lines_dict)
#print(  vezins_dict)

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

# hyper parameters
num_units = 128             
learning_rate = 0.001           
batch_size = 20
max_gradient_norm = 5.0
iter_size = 20000

inp_seq_length = len(enc_inp_num_data[0])
dec_inp_seq_length = len(dec_inp_num_data[0])
inp_alphabet_size = len(lines_dict)
out_alphabet_size = len(vezins_dict)
embedding_size = 10


#print("inp_seq_length : ", inp_seq_length, " dec_inp_seq_length : ", dec_inp_seq_length, " inp_alphabet_size : ", inp_alphabet_size, " : out_alphabet_size : ", out_alphabet_size)

# create batch iterator
iterator = p_data.NmtBatchIterator(enc_inp_num_data, dec_inp_num_data, target_num_data, batch_size)

"""
try:
    while True:
        _, _, i = iterator.next()
        print("shape ", np.shape(i))
except StopIteration:
    print("finished")
"""

TRAINING = "train"
INFERENCE = "infer"

def get_model(mode):
    """
        Creates model for training or inference.
        
    """
    reuse = False
    if mode == INFERENCE:
        reuse = True
        
    with tf.variable_scope('root', reuse=reuse):
    
        
        # Placeholders
        encoder_inputs = tf.placeholder(tf.int32, (inp_seq_length, batch_size), 'inputs')
        decoder_inputs = tf.placeholder(tf.int32, (dec_inp_seq_length, batch_size), 'outputs')
        targets = tf.placeholder(tf.int32, (dec_inp_seq_length, batch_size), 'targets')
        seqlen = tf.placeholder(tf.int32, [batch_size], 'seqlen')
        target_seqlen = tf.placeholder(tf.int32, [batch_size], 'target_seqlen')

        # Embedding
        embedding_encoder = tf.get_variable(
                "embedding_encoder", [inp_alphabet_size, embedding_size])
        encoder_emb_inp = tf.nn.embedding_lookup(
                embedding_encoder, encoder_inputs)

        if mode == TRAINING:
            embedding_decoder = tf.get_variable(
                    "embedding_decoder", [out_alphabet_size, embedding_size])
            decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder, decoder_inputs)
        else:
            embedding_decoder = tf.get_variable(
                "embedding_decoder", [out_alphabet_size, embedding_size])


        # Encoder graph
        # lstm cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # Dynamic RNN
        # encoder_outputs: [max_time, batch_size, num_units]
        # encoder_state: [batch_size, num_unis]
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_emb_inp, 
                sequence_length=seqlen, time_major=True, dtype=tf.float32)

        # Decoder graph
        # lstm cell 
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        
        if mode == TRAINING: # Training

            # decoder_lengths = np.empty(batch_size)
            # decoder_lengths.fill(dec_inp_seq_length, dtype=tf.int32)
            decoder_lengths = np.full((batch_size), dec_inp_seq_length, dtype=np.int32)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, decoder_lengths, time_major=True)
        else: # Inference

            # Greedy helper
            start_tokens = tf.tile([vezins_dict[start_token_str]], [batch_size])
            end_token = vezins_dict[end_token_str]

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_decoder, start_tokens=start_tokens, end_token=end_token)
            

        # Projection layer
        #projection_layer = layers_core.Dense(
        #       out_alphabet_size , use_bias=False) # the out_alphabet_size can be wrpng parameter
        projection_layer = layers_core.Dense(
            out_alphabet_size, use_bias=False)

        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state, 
            output_layer=projection_layer)



        if mode == TRAINING:    # Training

            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=dec_inp_seq_length, output_time_major=False)
            logits = outputs.rnn_output

            # Loss
            target_output = tf.transpose(targets)
            
            # adding padding
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_output, logits=logits)
            target_weights = tf.sequence_mask(
                target_seqlen, dec_inp_seq_length, dtype=logits.dtype)
            train_loss = (tf.reduce_sum(cross_entropy*target_weights) /
                batch_size)
            prediction = tf.argmax(logits, 2)
            # Model evaluation
            #correct_pred = tf.equal(tf.cast(prediction, tf.int32), targets)
            #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            #prediction = tf.transpose(p)
            #print("shape : ", prediction.shape)
            #train_loss = tf.losses.softmax_cross_entropy(
            #        onehot_labels=onehot_labels, logits)
            #train_loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, dec_inp_seq_length]))

            # Optimization
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))
            initializer = tf.global_variables_initializer()

        else: #Inference
            
            maximum_iterations = tf.round(tf.reduce_max(dec_inp_seq_length) * 2)
            # Dynamic decoding
            infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=dec_inp_seq_length, output_time_major=False)
            infer_logits = infer_outputs.rnn_output


            pad_size = dec_inp_seq_length - tf.shape(infer_logits)[1]
            infer_logits = tf.pad(infer_logits, [[0,0], [0, pad_size], [0,0]])

            logits_shape = tf.shape(infer_logits)


            """
            logits_stack = tf.reshape(
                    infer_logits,
                    shape=[-1, out_alphabet_size ])
            """
            # Loss
            infer_target_output = tf.transpose(targets)

            """
            current_batch_max_len = tf.shape(infer_logits)[1]

            target_seqs = tf.slice(
                    infer_target_output,
                    [0, 0],
                    [batch_size, current_batch_max_len])
            

            infer_targets_cross = tf.reshape(
                    infer_target_output,
                    shape=[-1])
            """
            #print("logits shape: ", tf.shape(infer_logits))

            #pad_size = 1260 - tf.shape(infer_logits)[0]
            #infer_logits = tf.pad(infer_logits, [[0,0], [0, pad_size], [0,0]]) 

            infer_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=infer_target_output, logits=infer_logits)
            infer_target_weights = tf.sequence_mask(
                target_seqlen, dec_inp_seq_length, dtype=infer_logits.dtype)
            infer_train_loss = (tf.reduce_sum(infer_cross_entropy*infer_target_weights) /
                batch_size)
            infer_prediction = tf.argmax(infer_logits, 2, name="prediction")
            saver = tf.train.Saver(max_to_keep=1000)

        if mode == TRAINING:
            
            return prediction, train_loss, update_step, initializer, encoder_inputs, decoder_inputs, targets, seqlen, target_seqlen

        else:
            return infer_prediction, infer_outputs, saver, encoder_inputs, seqlen

# create model
print("Creating models...")
# training
prediction, train_loss, update_step, initializer, encoder_inputs, decoder_inputs, targets, seqlen, target_seqlen = get_model(TRAINING)
# inference 
infer_prediction, infer_outputs, saver, infer_encoder_inputs, infer_seqlen = get_model(INFERENCE)

# logs
tf.summary.scalar("loss", train_loss)
# model saver
#saver = tf.train.Saver()

# Training

sess = tf.Session()

sess.run(initializer)

#for i in itertools.count():
    #train_input_data = ...
    #sess.run([loss, update_step], feed_dict={input_data: train_input_data})
# logging
timestamp = str(math.trunc(time.time()))
merged_summ = tf.summary.merge_all()
training_logger = tf.summary.FileWriter("./log/" + timestamp + "-training", sess.graph)
test_logger = tf.summary.FileWriter("./log/" + timestamp + "-testing", sess.graph)

print("Training...")

INFER_TIME = 100
for i in range(iter_size):
    try:
        encoder_inp, decoder_inp, trgt, seq_length, trgt_seq_length, t= iterator.next()

        # Training 
        pred, loss, _, train_summary= sess.run([prediction, train_loss, update_step, merged_summ], feed_dict={
            encoder_inputs:list(encoder_inp),
            decoder_inputs:list(decoder_inp),
            targets:list(trgt),
            seqlen:list(seq_length),
            target_seqlen:list(trgt_seq_length)})
            
        # log training summariies
        training_logger.add_summary(train_summary, i)


        if i%INFER_TIME == 0:
            print("---Train---")
            p_data.print_result(pred, t, vezins_reverse_dict)
            #print("Loss : ", loss, " predict : ", pred[0], " target : ", t[0])
            print("Loss : ", loss)
            encoder_inp, decoder_inp, trgt, seq_length, trgt_seq_length, target_print = iterator.test_next()

            pred, _= sess.run([infer_prediction, infer_outputs], feed_dict={
                       infer_encoder_inputs:list(encoder_inp),
                       infer_seqlen:list(seq_length)})
            print("---Test---")
            p_data.print_result(pred, t, vezins_reverse_dict)
            #print("Loss : ", loss, " predict : ", pred[0], " target : ", t[0])

        if i%1000 ==0:
            # save session
            saved_file = saver.save(sess, 'checkpoints/nmt_v2_train_' + timestamp, global_step=i)

            # log testing summaries
            #test_logger.add_summary(test_summary, i)
 
        

    except StopIteration:
        iterator.iter = 0
        


