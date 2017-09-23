import numpy


# viterbi[] = viterbi state chart
# observation_sequence = list containing sentence to be tagged
# state_sequence = list containing possible tags of sentence
# a[] = Transition Probability Matrix for the observation_sequence
# b[] = Emission Probability Matrix for the observation_sequence
# backtrace[] = stores the argmax for each state

def viterbi_decode(observation_sequence, state_sequence, a, b, pie_1, pie_2):

    rows = len(state_sequence)
    columns = len(observation_sequence)
    viterbi = numpy.zeros((rows, columns))
    backtrace = numpy.zeros((rows, columns))
    best_path = numpy.zeros(columns)

    # Initialization - we fill the first column with <s> to tag probabilities.

    for s, state in enumerate(state_sequence):
            viterbi[s][0] = pie_1[s]*b[s][0]
            backtrace[s][0] = 0

    # Recursion - we will fill the remaining columns in the state chart

    for t, observation in enumerate(observation_sequence):
        if t != 0:
            for s, state in enumerate(state_sequence):
                func = []
                func_max = []
                for s_iter, state_i in enumerate(state_sequence):
                    v1 = viterbi[s_iter][t-1]
                    v2 = a[s_iter][s]
                    v3 = b[s][t]
                    func.append(viterbi[s_iter][t-1]*a[s_iter][s]*b[s][t])
                    func_max.append(viterbi[s_iter][t - 1] * a[s_iter][s])
                viterbi[s][t] = numpy.amax(func)
                backtrace[s][t] = numpy.argmax(func_max)

    # Termination - we will fill the remaining columns in the state chart
    end_col = len(observation_sequence) - 1
    func_max = []
    for s_iter, state_i in enumerate(state_sequence):
        v1 = viterbi[s_iter][end_col]
        v2 = a[s_iter][s]
        func_max.append(viterbi[s_iter][end_col] * a[s_iter][s])

    best_score = numpy.amax(func_max)
    start_backtrace = numpy.argmax(func_max)

    # Backtracking

    best_path[end_col] = start_backtrace
    for index in range(1, -1, -1):
        use_index = int(best_path[index+1])
        best_path[index] = backtrace[use_index][index+1]

    return best_path


if __name__ == "__main__":

    os = ['Janet', 'will', 'back']
    ss = ['NNP', 'MD', 'VB']

    a_1 = numpy.zeros((3, 3))
    b_1 = numpy.zeros((3, 3))

    pie_1 = [0.28000, 0.00006, 0.00310]
    pie_2 = [0.01, 0.01, 0.01]

    a_1 = [[0.38000, 0.01100, 0.00090],
           [0.00008, 0.00020, 0.79680],
           [0.03220, 0.00050, 0.00500]]

    b_1 = [[0.00003, 0, 0],
           [0, 0.3080, 0],
           [0, 0.000028, 0.00067]]

    backtrace_list = viterbi_decode(os, ss, a_1, b_1, pie_1, pie_2)
    answer_string = []
    for index in range(0, len(os), 1):
        tag = backtrace_list[int(index)]
        answer_string.append(ss[int(tag)])

    print(answer_string)

