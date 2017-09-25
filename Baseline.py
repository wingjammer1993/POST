import csv


# This function will return the array of all tags with frequency count for user_input string

def train_freq_counts(filename, separator):
    freq_dict = {}
    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True):
            if line:
                input_word = line[1]
                input_word_tag = line[-1]
                tag_set = {}
                if input_word not in freq_dict:
                    freq_dict[input_word] = tag_set
                    tag_set[input_word_tag] = 1
                else:
                    tags_list = freq_dict[input_word]
                    if input_word_tag in tags_list:
                        tags_list[input_word_tag] = tags_list[input_word_tag]+1
                    else:
                        tags_list[input_word_tag] = 1
    return freq_dict


def print_baseline_output(training, filename, separator):
    freq = train_freq_counts(training, separator)
    with open("output.txt", "r+", newline='') as output:
        with open(filename, "r") as get_input:
            writer = csv.writer(output, delimiter=separator, skipinitialspace=True)
            for row in csv.reader(get_input, delimiter=separator, skipinitialspace=True):
                if row:
                    input_word = row[1]
                    if len(input_word) != 0:
                        if input_word in freq:
                            tag_dict = freq[input_word]
                            pos_tag = max(tag_dict, key=lambda key: tag_dict[key])
                            writer.writerow(row + [pos_tag.strip()])
                        else:
                            writer.writerow(row + ['NNP'.strip()])
                    else:
                        writer.writerow('\n'.strip())
                else:
                    writer.writerow('\n'.strip())


