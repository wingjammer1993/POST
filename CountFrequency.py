import re
import operator
import csv


def give_freq_counts(filename, separator, column):
    with open(filename, 'r') as file_obj:
        frequency = {}
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True):
            if line:
                input_word = line[column]
                if input_word in frequency:
                    count = frequency.get(input_word, 0)
                    frequency[input_word] = count + 1
                else:
                    frequency[input_word] = 1

    sorted_x = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x)
    return sorted_x


if __name__ == "__main__":
    output_file = give_freq_counts(r'Training_Berp.txt', '\t', 1)
    print("success")