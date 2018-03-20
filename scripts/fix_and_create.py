
import os

data_dir = "../data_edited/"
new_data_dir = "../data_fixed/"


for root, dirs, files in os.walk(data_dir):
    path = root.split(os.sep)
    for file in files:
        with open(data_dir + path[2] + "/" + str(file), "r") as f:
            new_file = open(new_data_dir + path[2] + "/" + str(file), "w")
            for line in f:
                for c in line:
                    if c == "é":
                        new_file.write("e")
                    elif c == "’" :
                        new_file.write("'")
                    elif c == "ñ" :
                        new_file.write("n")
                    elif c == "‘" :
                        new_file.write("'")
                    else :
                        new_file.write(c)
            new_file.close()
 
