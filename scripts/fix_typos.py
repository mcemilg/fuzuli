
import os

data_dir = "../data_fixed/"


for root, dirs, files in os.walk(data_dir):
    path = root.split(os.sep)
    for file in files:
        with open(data_dir + path[2] + "/" + str(file), "r") as f:
            for line in f:
                for c in line:
                    if c == "‘":
                        print ( data_dir + path[2] + "/" + str(file))
        
