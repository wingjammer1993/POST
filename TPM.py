import csv
import nltk
import CountFrequency


# This method is parsing the column of given filename file and constructing a sentence with start
# and end markers. The complete file is parsed.
# may further develop method to include row parameter to partially parse the file


def construct_sentence_sequence(filename, separator, column, row):
    with open(filename, 'r') as file_obj:
        sentence = ['<s>']
        for row, line in enumerate(csv.reader(file_obj, delimiter=separator, skipinitialspace=True)):
            if line:
                input_word = line[column]
                sentence.append(input_word)
            else:
                sentence.append('<e>')
                sentence.append('<s>')
    return sentence


# This method is generating bigram tuples from the parsed file, using nltk

def get_bigrams(sentence):
    bigram = list(nltk.bigrams(sentence))
    return bigram


# This method is taking the bigram list, unigram counts, and the filename.
# For every tag in the filename (tag list), it is finding out how many times that tag is followed by another tag
# It then arranges this data in a tag x tag matrix
# It returns a list of dictionaries, each dictionary corresponds to a column of transition probability matrix
# dictionaries are ordered according to tag list

def get_tag_bigram_probability(filename, bigram_list, unigram_count):
    tag_dictionary_list = {}
    with open(filename, encoding="utf8") as file_obj:
        content = file_obj.readlines()
        for tag_name_row in content:
            if tag_name_row:
                tag_successors = [b for (a, b) in bigram_list if a == tag_name_row.strip()]
                freq_dist = nltk.FreqDist(tag_successors)
                unigram_tag_count = unigram_count.get(tag_name_row.strip())
                with open(filename, encoding="utf8") as re_file_obj:
                    tag_tag_dictionary = {}
                    for tag_name_column in re_file_obj.readlines():
                        if tag_name_column:
                            tag_bigram_count = freq_dist[tag_name_column.strip()]
                            if unigram_tag_count:
                                tag_tag_dictionary[tag_name_column.strip()] = tag_bigram_count/unigram_tag_count
                            else:
                                tag_tag_dictionary[tag_name_column.strip()] = 0
                    tag_dictionary_list[tag_name_row.strip()] = tag_tag_dictionary

    return tag_dictionary_list


def get_initial_pi_matrix(filename, bigram_list, unigram_count):
    tag_successors = [b for (a, b) in bigram_list if a == '<s>']
    freq_dist = nltk.FreqDist(tag_successors)
    unigram_tag_count = unigram_count.count('<s>')
    with open(filename, encoding="utf8") as re_file_obj:
        tag_tag_dictionary = {}
        for tag_name_column in re_file_obj.readlines():
            if tag_name_column:
                tag_bigram_count = freq_dist[tag_name_column.strip()]
                if unigram_tag_count:
                    tag_tag_dictionary[tag_name_column.strip()] = tag_bigram_count / unigram_tag_count
                else:
                    tag_tag_dictionary[tag_name_column.strip()] = 0
    return tag_tag_dictionary


def get_end_pi_matrix(filename, bigram_list, unigram_count):
    tag_dictionary_list = {}
    with open(filename, encoding="utf8") as file_obj:
        content = file_obj.readlines()
        for tag_name_row in content:
            if tag_name_row:
                tag_successors = [b for (a, b) in bigram_list if a == tag_name_row.strip()]
                freq_dist = nltk.FreqDist(tag_successors)
                unigram_tag_count = unigram_count.get(tag_name_row.strip())
                tag_bigram_count = freq_dist['<e>']
                tag_tag_dictionary = {}
                if unigram_tag_count:
                    tag_tag_dictionary['<e>'] = tag_bigram_count / unigram_tag_count
                else:
                    tag_tag_dictionary['<e>'] = 0
                tag_dictionary_list[tag_name_row.strip()] = tag_tag_dictionary
    return tag_dictionary_list


# This method creates a csv of the TPM matrix

# def create_csv(data_list, output_filename):
#    keys = data_list.keys()
#    with open(output_filename, 'w') as output_file:
#        dict_writer = csv.DictWriter(output_file, keys, delimiter='\t')
#        dict_writer.writerows(data_list.keys())


# This method calls all other methods

def get_transition_probability_matrix():
    sentence_sequence_list = construct_sentence_sequence(r'Training_Berp.txt', '\t', 2, 0)
    tag_tag_pairs = get_bigrams(sentence_sequence_list)
    tag_frequency_count = CountFrequency.give_freq_counts(r'Training_Berp.txt', '\t', 2)
    tag_bigram_count_dictionary = get_tag_bigram_probability(r'POSTagList.txt', tag_tag_pairs, tag_frequency_count)
#    create_csv(tag_bigram_count_dictionary, 'TPM_out.csv')
#    sentence_sequence_word_list = construct_sentence_sequence(r'Training_Berp.txt', '\t', 1, 0)
#    initial_pi_matrix = get_initial_pi_matrix(r'POSTagList.txt', tag_tag_pairs, sentence_sequence_word_list)
#    end_pi_matrix = get_end_pi_matrix(r'POSTagList.txt', tag_tag_pairs, tag_frequency_count)
#    print(initial_pi_matrix)
#    print(end_pi_matrix)
    print(tag_bigram_count_dictionary)
    print('success')
    return tag_bigram_count_dictionary


if __name__ == "__main__":
    get_transition_probability_matrix()
