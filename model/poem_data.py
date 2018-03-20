#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import numpy as np
import collections
from random import shuffle
from sklearn.utils import shuffle as sklearn_shuffle

'''
    File: poem_data.py
    Author: M.Cemil Guney
    Created Date: 09.11.2017
    Description: This module utils for reading and manuplating the poems data.
'''

'''
    The Poem Data
    - line
        - { 'l', 'i', 'n', 'e', ' ', 'c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', 's' }
    
    - vezin
        - { 'F', 'a', 'i', 'l', 'a', 't', 'ü', 'n', ' ', 'F', 'a', 'i', 'l', 'ü', 'n' }
    
    - vezins
        - { 1 : vezin, 2 : another_vezin ... }

    - lines
        - { 1 : [line, another_line], 2 : [yet_another_one, ...] ... }
        - The numbers are vezin numbers 
            - vezins[1] => vezin


'''

'''
    The data_path folder design
    - data_path = "/some/path/poem"
        - 1 (Folder)
            - 1 (Poem File)
            - 2 (Poem File)
            - 3 (Poem File)
        - 2 (Folder)
            - 1 (Poem File)
            - 2 (Poem File)
        - vezin_list (File)
            (Content)
            Mefâilün Mefâilün Feûlün
            Mef'ûlü Mefâîlü Mefâîlü Feûlûn
            
            (Each line represents its folder number. Like the first 
             lines poems is under the 1 folder)
'''


def read_all_poems(data_path):
    """
        Reads all poems under the data_path respect by the folder design.
        returns the vezins and lines respect by the poem data desgin.
    """
    vezin_file = data_path + "/vezin_list"
    
    vezins = read_vezin_file(vezin_file)

    # this variable for creating list for each vezin by number
    lines = {}
    for key, vezin in vezins.items():
        #print("line : ", line, " vezin ", vezin)
        poem_path = data_path + "/" + str(key) + "/"
        for f in os.listdir(poem_path):
            poem = read_poem(poem_path + f)

            if key not in lines:
                lines[key] = []
            lines[key].extend(poem)
                
            #print(lines)
    
    return vezins, lines 

def read_vezin_file(fname):
    """
        Reads vezin_list file and returns vezins dictionary by line 
        with respect to the file design.
    """
    with open(fname) as f:
        content = f.readlines()
    
    content = [x.strip() for x in content]  
    content = [list(x) for x in content]
    
    vezins = {}
    line = 1
    for c in content:
        vezins[line] = c
        line += 1

    return vezins

def read_poem(fname):
    """
        Read poem file and returns it line list by char.
    """

    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content if x != []]
    #content = [content[i].split() for i in range(len(content))]
    content = [list(x) for x in content]
    content = [x for x in content if x != []]
    #content = np.array(content)
    #content = np.reshape(content, [-1, ])
    
    return content

def build_dictionary(lines, vezins, key="lines"):
    """
        Creates a dictionary which contains integer provisions of chars of
        lines or vezins.
        The provisions created order by most common.
        Returns dictionary and also the reverse.
    """
   
    # get only lines chars
    lines_chars, vezins_chars = append_all_chars(lines, vezins)
    chars = []
    if key == "lines":
        chars = lines_chars
    elif key == "vezins":
        chars = vezins_chars

    #chars = list(itertools.chain.from_iterable(words)) 
    count = collections.Counter(chars).most_common()
    dictionary = dict()
    for char, _ in count:
        dictionary[char] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary

def add_key_to_dict(dictionary, reverse_dictionary, attribute, key="max"):
    
    if key == "max":
        max_val = 0
        for key, value in dictionary.items():
            if value > max_val:
                max_val = value
        max_val += 1
        dictionary[attribute] = max_val
        reverse_dictionary[max_val] = attribute
    else:
        # todo: add value with given key
        pass

    return dictionary, reverse_dictionary

def append_all_chars(lines, vezins):
    """
        Gets lines and vezins as input and 
        creates lines_chars and vezins_chars array
    """
    
    lines_chars = []
    vezins_chars = []

    for key, vezin in vezins.items():
        vezins_chars.extend(vezin)
        for line in lines[key]:
            lines_chars.extend(line)

    return lines_chars, vezins_chars


def create_batch(lines):
    """
        Creates a batch from lines. The lines needs to be respect format.
        The returned batch will be shuffled.
    """

    train_batch = []
    test_batch = []
    
    # create a list form lines dictionary
    for key, items in lines.items():

        # get the 95 percent of lines as train
        train_length = len(items)*95//100
        
        for i in range(0, train_length):
            train_batch.append([key, items[i]])
            
        for i in range(train_length, len(items)):
            test_batch.append([key, items[i]])
    
    # this shuffling not healthful
    shuffle(train_batch)
    shuffle(test_batch)

    return test_batch, train_batch


def nmt_create_data(lines, vezins):
    """
        Creates data from lines and vezins for seq2seq model.
        Datas will be padded.
        Returns:
            encoder_input_data,
            decoder_input_data, 
            target_data
    """
    
    line_max_length = get_max_length(lines, "lines")
    vezin_max_length = get_max_length(vezins, "vezins") +1 # +1 for start and end char
    
    encoder_input_data = []
    decoder_input_data = []
    target_data = []

    for key, items in lines.items():
        # decoder input
        # [ '<go>', 'F', 'i', ... , 'n', <pad>, <pad>]
        decoder_input = vezins[key][:] # copy vezin
        decoder_input = ['<go>'] + decoder_input
        decoder_input = padding_vector(decoder_input, vezin_max_length, padding=" ")

        # target
        # [ 'F', 'i', ... , 'n',</go>, <pad>, <pad>]
        target = vezins[key][:]
        target = target + ['</go>']
        target = padding_vector(target, vezin_max_length, padding=" ")

        for line in items:
            decoder_input_data.append(decoder_input)
            target_data.append(target)
            encoder_input = padding_vector(line, line_max_length, padding=" ")
            encoder_input_data.append(encoder_input)

    #print("decoder_input : ", decoder_input_data[0])
    #print("encoder_input : ", encoder_input_data[0])
    #print("target : ", target_data[0])

    return encoder_input_data, decoder_input_data, target_data

def nmt_create_poem_generation_data_v1(lines, vezins):
    """
        Creates data for poem generation.
        Returns: 
            encoder_input_data : lines for encoder input
            decoder_input_data : vezins for decoder input
            target_data        : targets are second lines of input lines
    """


    line_max_length = get_max_length(lines, "lines")
    vezin_max_length = get_max_length(vezins, "vezins") +1 # +1 for start and end char
    encoder_input_data = []
    decoder_input_data = []
    target_data = []

    for key, items in lines.items():
        # decoder input
        # [ '<go>', 'F', 'i', ... , 'n', <pad>, <pad>]
        decoder_input = vezins[key][:] # copy vezin
        decoder_input = ['<go>'] + decoder_input
        decoder_input = padding_vector(decoder_input, vezin_max_length, padding=" ")

        # iterate the lines to before last element
        for i in range(len(items)-1):
            first_line = items[i]
            second_line = items[i+1]
            # decoder input
            decoder_input_data.append(decoder_input) 

            # encoder input
            encoder_input = padding_vector(first_line, line_max_length, padding=" ")
            encoder_input_data.append(encoder_input)

            target = second_line[:]
            target = target + ['</go>']
            # the max length vezin length because the decoder
            # inputs max length is vezins max length
            target = padding_vector(target, vezin_max_length, padding=" ")
            target_data.append(target)


    print("decoder_input : ", decoder_input_data[0])
    print("encoder_input : ", encoder_input_data[0])
    print("target : ", target_data[0])

    return encoder_input_data, decoder_input_data, target_data

def nmt_create_poem_generation_data_v2(lines, vezins):
    """
        Creates data for poem generation.
        Returns: 
            encoder_input_data : lines for encoder input
            decoder_input_data : second line for decoder input
            target_data        : second line for target
    """


    line_max_length = get_max_length(lines, "lines")
    encoder_input_data = []
    decoder_input_data = []
    target_data = []

    for key, items in lines.items():

        # iterate the lines to before last element
        for i in range(len(items)-1):
            first_line = items[i]
            second_line = items[i+1]

            # encoder input
            encoder_input = padding_vector(first_line, line_max_length, padding=" ")
            encoder_input_data.append(encoder_input)

            # decoder input
            decoder_input = second_line[:]
            decoder_input = ['<go>'] + decoder_input
            decoder_input = padding_vector(decoder_input, line_max_length, padding=" ")
            decoder_input_data.append(decoder_input)

            
            target = second_line[:]
            target = target + ['</go>']
            target = padding_vector(target, line_max_length, padding=" ")
            target_data.append(target)


    print("decoder_input : ", decoder_input_data[0])
    print("encoder_input : ", encoder_input_data[0])
    print("target : ", target_data[0])

    return encoder_input_data, decoder_input_data, target_data




def convert_to_num(data, dictionary):
    """
        Convert data to number equivalent at the dictionary.
        Returns converted data.
    """

    converted_data = []
    
    for item in data:
        converted = []
        for c in item:
            converted.append(dictionary[c])
        converted_data.append(converted)

    return converted_data
    

def get_max_length(data, data_type):
    """
        Finds max length of lines or vezins.
        data_type : "lines"  for lines
                    "vezins" for vezins
        returns: 
                max_length
    """

    max_length = 0

    if data_type == "lines":
        for key, lines in data.items():
            for l in lines:
                if max_length < len(l):
                    max_length = len(l)
    elif data_type == "vezins" :
        for _, v in data.items():
            if max_length < len(v):
                max_length = len(v)

    return max_length
    

'''
def create_multiline_batch(lines):
    
        Creates a multiline(2 lines) batch.
        Returned batch will be shuffled.
    

    train_batch = []
    test_batch = []

    for key, items in lines.items():

        # get the 80 percent of
'''

def padding_vector(vector, max_length, padding=0):
    """
        Adds padding to the vector for expanding to the max length.
        Returns resized vector.
    """
    
    padding_vec = []
    padding_vec.extend(vector)
    
    for i in range(max_length - len(vector)):
        padding_vec.append(padding)


    return padding_vec

def print_result(predictions, targets, dictionary):

    print(" -Predictions- ")
    for p in predictions:
        pred = ""
        for i in p:
            pred += dictionary[i]
            
        print(pred)
    
    print(" -Targets- ")
    for t in targets:
        target = ""
        for i in t:
            target += dictionary[i]
            
        print(target)






class NmtBatchIterator:
    """
        Simple helper class for iterateion of batches for nmt model.
    """

    def __init__(self, encoder_inp_data, decoder_inp_data, target_data, batch_size):
        self.batch_size = batch_size
        # shuffle the data
        shuffled_enc_inp, shuffled_dec_inp, shuffled_target = sklearn_shuffle(encoder_inp_data, decoder_inp_data, target_data)
        
        # % 80 train , %20 train
        self.train_length = len(shuffled_enc_inp)*80//100

        #print("train_length : ", self.train_length)
        
        self.train_enc_inp = shuffled_enc_inp[:self.train_length]
        self.test_enc_inp = shuffled_enc_inp[self.train_length:]
        
        self.train_dec_inp = shuffled_dec_inp[:self.train_length]
        self.test_dec_inp = shuffled_dec_inp[self.train_length:]
        
        self.train_target = shuffled_target[:self.train_length]
        self.test_target = shuffled_target[self.train_length:]

        #self.sequence_length = slef.create_sequence_length(self.train_enc_inp, self.batch_size)

        # iterator
        self.iter = 0
        self.test = 0

    def next(self):
        """
            Returns next batch data. The data will be transpose of the lines.
        """


        self.iter += 1
        if self.iter*self.batch_size < self.train_length:
            start = (self.iter-1)*self.batch_size 
            end = self.iter*self.batch_size
            return self.transpose(self.train_enc_inp[start:end]), \
                    self.transpose(self.train_dec_inp[start:end]),\
                    self.transpose(self.train_target[start:end]),\
                    self.get_seqlen(self.train_enc_inp[start:end]),\
                    self.get_seqlen(self.train_target[start:end]),\
                    self.train_target[start:end]

        elif (self.iter-1)*self.batch_size < self.train_length:
            raise StopIteration()
            """
            start = (self.iter-1)*self.batch_size
            return self.transpose(self.train_enc_inp[start:]),\
                    self.transpose(self.train_dec_inp[start:]),\
                    self.transpose(self.train_target[start:])
            """
        else:
            raise StopIteration()
    
    def test_next(self):
        start = 0
        end = self.batch_size 
        return self.transpose(self.test_enc_inp[start:end]), \
                        self.transpose(self.test_dec_inp[start:end]),\
                        self.transpose(self.test_target[start:end]),\
                        self.get_seqlen(self.test_enc_inp[start:end]),\
                        self.get_seqlen(self.test_target[start:end]),\
                        self.test_target[start:end]


    """
    def create_sequence_length(self, sequence, batch_sz):
        sequence_length = []
        
        for i in range(batch_sz):
            sequence_length.append(
    """

    def transpose(self, data):
        return map(list, zip(*data))

    def get_seqlen(self, data):
        """
            Calculates the exact sequence length without padding.
        """
        seqlen = []
        for arr in data:
            temp = arr[:]
            for i in range(len(data)):
                if temp[len(temp)-1] == ' ':
                    temp = temp[:-1]
                else:
                    break
            seqlen.append(len(temp)) 

        return seqlen


