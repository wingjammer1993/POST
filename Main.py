import TPM
import EPM
import ViterbiDecoding
import LocalParamters
import CountFrequency
import Input_Generation


def fetch_output():

    # Defines

    tags_file = r'POSTagList.txt'
    training_file = r'Training_Berp.txt'
    dev_set = r'NEW_EVAL_TASK.txt'
    output_file = 'output.txt'
    delimiter = '\t'
    word_column = 1
    tag_column = 2

    # Initialization

    answers = []

    # Parsing sentences from training data and generating word-tag bigrams

    tag_frequency_count = CountFrequency.give_freq_counts(training_file, delimiter, tag_column)
    word_frequency_count = CountFrequency.give_freq_counts(training_file, delimiter, word_column)
    sentence_sequence_word_list = TPM.construct_sentence_sequence(training_file, delimiter, word_column, 0)
    sentence_sequence_tag_list = TPM.construct_sentence_sequence(training_file, delimiter, tag_column, 0)
    unked_sequence_word_list = Input_Generation.define_training_unk_words(word_frequency_count, sentence_sequence_word_list)
    word_tag_pairs = EPM.get_epm_bigrams(sentence_sequence_tag_list, unked_sequence_word_list)
    tag_tag_pairs = TPM.get_bigrams(sentence_sequence_tag_list)
    vocabulary = set(unked_sequence_word_list)
    # Creating the master parameter list

    master_a = TPM.get_transition_probability_matrix(tags_file, tag_tag_pairs, tag_frequency_count)
    master_b = EPM.get_emission_probability_matrix(tags_file, vocabulary, word_tag_pairs, tag_frequency_count)
    master_pie_1 = TPM.get_initial_pi_matrix(tags_file, tag_tag_pairs, sentence_sequence_word_list)
    master_pie_2 = TPM.get_end_pi_matrix(tags_file, tag_tag_pairs, tag_frequency_count)

    # Generating the list of sentences to be fed

    all_inputs = TPM.construct_sentence_sequence(dev_set, delimiter, 1, 0)

    # Find out the state_sequence_list and observation_sequence_list

    extracted_inputs = Input_Generation.extract_input_sentences_list(all_inputs)
    unked_extracted_inputs = Input_Generation.define_extracted_unk_words(unked_sequence_word_list, extracted_inputs)
    extracted_tags = Input_Generation.extract_possible_tags_list(extracted_inputs, word_tag_pairs)

    # loop this in for all sentences
    for index, observation_sequence in enumerate(unked_extracted_inputs):

        state_sequence = extracted_tags[index]

        # construct matrix A
        a = LocalParamters.construct_local_transition(state_sequence, master_a)

        # construct matrix B
        b = LocalParamters.construct_local_emission(state_sequence, observation_sequence, vocabulary, master_b)

        # construct matrix pie_1
        pie_1 = LocalParamters.construct_local_pie_start(state_sequence, master_pie_1)

        # construct matrix pie_2
        pie_2 = LocalParamters.construct_local_pie_end(state_sequence, master_pie_2)

        # input it to viterbi
        answer_string = ViterbiDecoding.viterbi_decode(observation_sequence, state_sequence, a, b, pie_1, pie_2)

        # fetch the answer strings
        answers.append(answer_string)


if __name__ == "__main__":
    fetch_output()