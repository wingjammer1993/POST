

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





