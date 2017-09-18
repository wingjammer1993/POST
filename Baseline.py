import csv

# This function will return the array of all tags with frequency count for user_input string


def give_freq_counts(filename, separator, user_input):
    tag_set = [0] * 45
    with open(filename, 'r') as file_obj:
        for line in csv.reader(file_obj, delimiter=separator, skipinitialspace=True):
            if line:
                input_word = line[1]
                if user_input == input_word:
                    post_tag = line[-1]
                    num_tag = give_tag_info(post_tag)
                    if 0 <= tag_set[num_tag]:
                        tag_set[num_tag] = tag_set[num_tag] + 1
    return tag_set

# This function will return tag name given tag number and vice-versa
# Tags are read from POSTagList.txt


def give_tag_info(pos_tag_elem):
    with open('POSTagList.txt') as f:
        content = f.readlines()
        tag_num = -1
        if int == type(pos_tag_elem):
            tag_num = content[pos_tag_elem]
        if str == type(pos_tag_elem):
            for index, elem in enumerate(content):
                if pos_tag_elem == content[index].strip():
                    tag_num = index
                    break

    return tag_num


def get_input_file(filename, separator):
    with open("output.txt", "r+", newline='') as output:
        with open(filename, "r") as get_input:
            writer = csv.writer(output, delimiter=separator, skipinitialspace=True)
            for row in csv.reader(get_input, delimiter=separator, skipinitialspace=True):
                if row:
                    input_word = row[1]
                    print(input_word)
                    freq = give_freq_counts(r'Training_Berp.txt', '\t', input_word)
                    max_freq = max(freq)
                    if len(input_word) != 0:
                        print(freq)
                        pos_tag_num = freq.index(max_freq)
                        print(max_freq)
                        pos_tag = give_tag_info(pos_tag_num)
                        print(pos_tag)
                        writer.writerow(row + [pos_tag.strip()])
                    else:
                        writer.writerow('\n'.strip())
                else:
                    writer.writerow('\n'.strip())
    return output


if __name__ == "__main__":
    output_file = get_input_file(r'NEW_EVAL_TASK.txt', '\t')
    print("success")
