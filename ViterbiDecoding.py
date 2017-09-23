import numpy


# viterbi[] = viterbi state chart
# observation_sequence = list containing sentence to be tagged
# state_sequence = list containing possible tags of sentence
# a[] = Transition Probability Matrix for the observation_sequence
# b[] = Emission Probability Matrix for the observation_sequence
# backtrace[] = stores the argmax for each state

def viterbi_decode(observation_sequence, state_sequence, a, b):

    rows = len(state_sequence)
    columns = len(observation_sequence)
    viterbi = numpy.zeros((rows, columns))
    backtrace = numpy.zeros((rows, columns))
    best_path = numpy.zeros(columns)

    # Initialization - we fill the first column with <s> to tag probabilities.

    for s, state in enumerate(state_sequence):
            viterbi[s][0] = a[0][s]*b[s][0]
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
                backtrace[s][t] = numpy.argmax(func_max) + 1

    # Termination - we will fill the remaining columns in the state chart

    func = []
    func_max = []
    final_state = len(state_sequence)
    for s_iter, state_i in enumerate(state_sequence):
        v1 = viterbi[s_iter][columns - 2]
        v2 = a[s_iter][final_state-1]
        func.append(viterbi[s_iter][columns - 2] * a[s_iter][final_state-1])
    viterbi[final_state-1][columns-1] = numpy.amax(func)
    backtrace[final_state-1][columns-1] = numpy.argmax(func)

    for t in range(0, columns):
        best_path[t] = viterbi[:, t].argmax()

    return best_path


if __name__ == "__main__":

    os = ['Janet', 'will', 'back']
    ss = ['NNP', 'MD', 'VB']
    a_1 = numpy.zeros((3, 3))
    b_1 = numpy.zeros((3, 3))
    a_1 = [[0.28000, 0.00006, 0.00310],
           [0.38000, 0.01100, 0.00090],
           [0.00008, 0.00020, 0.79680],
           [0.03220, 0.00050, 0.00500]]

    b_1 = [[0.00003, 0, 0],
           [0, 0.3080, 0],
           [0, 0.000028, 0.00067]]

    backtrace_matrix = viterbi_decode(os, ss, a_1, b_1)

