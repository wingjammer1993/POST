import numpy


# viterbi[] = viterbi state chart
# observation_sequence = list containing sentence to be tagged
# state_sequence = list containing possible tags of sentence
# a[] = Transition Probability Matrix for the observation_sequence
# b[] = Emission Probability Matrix for the observation_sequence
# backtrace[] = stores the argmax for each state

def viterbi_decode(observation_sequence, state_sequence, a, b):

    rows = len(state_sequence) + 2
    columns = len(observation_sequence)
    viterbi = numpy.zeros((rows, columns))
    backtrace = numpy.zeros((rows, columns))

    # Initialization - we fill the first column with <s> to tag probabilities.
    start, stop = 1, len(state_sequence)-1
    for itern, state in enumerate(state_sequence):
        if start <= itern <= stop:
            viterbi[itern-1, 0] = a[itern-1, 1]*b[itern-1, 1]
            backtrace[itern-1, 0] = 0

    # Recursion - we will fill the remaining columns in the state chart
    # s = itern
    # t = index

    begin, end = 2, len(observation_sequence) - 1
    start, stop = 1, len(state_sequence)-1
    for index, observation in enumerate(observation_sequence):
        if begin <= index <= end:
            for itern, state in enumerate(state_sequence):
                if start <= itern < stop:
                    viterbi[itern - 1, index-1] = numpy.amax(viterbi[:, index-1]*a[:, itern-1]*b[itern-1, index-1])
                    func_max = viterbi[:, index-2]*a[:, itern-1]
                    backtrace[itern - 1, index-1] = numpy.argmax(func_max)

    # Termination - we will fill the remaining columns in the state chart

    viterbi[stop, end] = numpy.amax(viterbi[:, end] * a[:, stop])
    func_max = viterbi[:, end] * a[:, stop]
    backtrace[stop, end] = numpy.argmax(func_max)

    return backtrace

