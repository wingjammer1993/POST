import TPM
import EPM
import numpy
import ViterbiDecoding

def extract_input_sentences_list(all_inputs):
    extracted_input = []
    observation_sequence = []
    for index, word in enumerate(all_inputs):
        if word != '<s>':
            if word != '<e>':
                observation_sequence.append(all_inputs[index])
            else:
                extracted_input.append(observation_sequence)
                observation_sequence = []

    return extracted_input


def extract_possible_tags_list(extracted_sentences, i_word_tag_pairs):
    extracted_possible_tags = []
    extracted_tag_sublist = []
    for sublist in extracted_sentences:
        for element in sublist:
            for tag in i_word_tag_pairs:
                if element == tag[1]:
                    if tag[0] not in extracted_tag_sublist:
                        extracted_tag_sublist.append(tag[0])
        extracted_possible_tags.append(extracted_tag_sublist)
        extracted_tag_sublist = []
    return extracted_possible_tags


if __name__ == "__main__":

    # Generating word-tag bigram list & parse sentences

    sentence_sequence_word_list = TPM.construct_sentence_sequence(r'Training_Berp.txt', '\t', 1, 0)
    sentence_sequence_tag_list = TPM.construct_sentence_sequence(r'Training_Berp.txt', '\t', 2, 0)
    word_tag_pairs = EPM.get_epm_bigrams(sentence_sequence_tag_list, sentence_sequence_word_list)
    all_inputs = TPM.construct_sentence_sequence(r'NEW_EVAL_TASK.txt', '\t', 1, 0)

    # Find out the state_sequence_list and observation_sequence_list
    extracted_inputs = extract_input_sentences_list(all_inputs)
    extracted_tags = extract_possible_tags_list(extracted_inputs,word_tag_pairs)

    master_a = TPM.get_transition_probability_matrix()
    master_b = EPM.get_emission_probability_matrix()

    answers = []
    state_sequence = []
    observation_sequence = []

    # loop this in for all sentences

    # construct matrix A
    a = numpy.zeros(len(state_sequence), len(state_sequence))
    a = construct_local_transition(state_sequence, master_a)
    # construct matrix B
    b = numpy.zeros(len(state_sequence), len(observation_sequence))
    b = construct_local_emission(state_sequence,observation_sequence, master_b)

    # construct matrix pie_1
    pie_1 = numpy.zeros(len(state_sequence))
    pie_1 = construct_local_pie_start(state_sequence, master_a)
    # construct matrix pie_2
    pie_2 = numpy.zeros(len(state_sequence))
    pie_2 = construct_local_pie_end(state_sequence, master_a)

    # input it to viterbi
    answer_string = ViterbiDecoding.viterbi_decode(observation_sequence, state_sequence, a, b, pie_1, pie_2)

    # fetch the answer strings
    answers.append(answer_string)



