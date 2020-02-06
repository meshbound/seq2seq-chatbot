import numpy as np
import csv

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

path = ""
files = ["{}___.csv", "{}___.csv", "{}___.csv", "{}___.csv", "{}___.csv", "{}___.csv"]

newFile = open(path+"/data.txt", "w+")

def filter_line(line, whitelist):
    line = line.replace("r/", "")
    return ''.join([ch for ch in line if ch in whitelist])

def array_to_lower(array):
    new_array = []

    for line in array:
        new_array.append(line.lower())

    return new_array

for file in files:
    with open(file.format(path), 'r', encoding="UTF-8") as raw:
        reader = csv.reader(raw, delimiter=",")

        headers = next(reader)
        data = np.array(list(reader))
        content = array_to_lower([item[3] for item in data])

        lines = [filter_line(line, EN_WHITELIST) for line in content]

        total = 0
        for line in lines:
            print(line + " | " + str(total))
            if not line.isspace() and len(line) > 0:
                if not (("https" or "http" or "Joined the server.") in line):
                    newFile.write(line + "\n")
                else:
                    newFile.write("\n")
            total += 1
        raw.close()
newFile.close()
