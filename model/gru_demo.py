
import tensorflow as tf
import numpy as np
import gru_data as g_data

'''
    File: gru_demo.py
    Author: M.Cemil Guney
    Description: Generetes poems with gru model.
    Resource: https://github.com/martin-gorner/tensorflow-rnn-shakespeare
'''

# these must match what was saved !
ALPHASIZE = g_data.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

# memorized data
#check_point = "checkpoints/rnn_train_1513602234-105000000"

# demo
check_point = "checkpoints/rnn_train_1513861144-3000000"

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1513861144-3000000.meta')
    new_saver.restore(sess, check_point)
    x = g_data.char_dict['L']
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

        c = g_data.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = g_data.reverse_char_dict[c]
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0


