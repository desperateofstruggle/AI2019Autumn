import numpy
import scipy.stats

'''
function declaration: the iterator function called by emMainProcess to run one round computation
Parameters:
    probability - the probability of the head side in A and B [pA, pB]
    samples     - the sample set to be used
Returns:
    [newProbabilityOfHeadSideInA, newProbabilityOfHeadSideInB]
'''

def iterator(probability, samples):
    # to record the sum of each incident under two different (A & B)
    sums = [[0, 0], [0, 0]]

    for sample in samples:
        # compute the len of Head and the len of Tail
        lenSample = len(sample)
        sumH = 0
        for n in sample:
            if n == 1:
                sumH += n
        sumT = lenSample - sumH

        # compute the relative proportion of A and B (binary distribution with n = sumH, times = lenSample, p)
        binaryNormalA = scipy.stats.binom.pmf(sumH, lenSample, probability[0])
        binaryNormalB = scipy.stats.binom.pmf(sumH, lenSample, probability[1])

        # compute the result proportion
        wA = binaryNormalA / (binaryNormalA + binaryNormalB)
        wB = binaryNormalB / (binaryNormalB + binaryNormalA)

        # update the sum of each incident under two different (A & B)
        sums[0][0] += wA * sumH
        sums[0][1] += wA * sumT
        sums[1][0] += wB * sumH
        sums[1][1] += wB * sumT

    return [sums[0][0] / (sums[0][0] + sums[0][1]), sums[1][0] / (sums[1][0] + sums[1][1])]


'''
function declaration: the main process based on em algorithm to solve the problem
Parameters:
    samples         - the sample set to be used
    probability     - the Hypothesis probabilities of the head side in A and B [pA, pB]
    mostIterations  - the largest rounds to iterate
    threshold       - the threshold to weigh the varieties of the probabilities between two iteration rounds
    
Returns:
    [resultProbabilityOfHeadSideInA, resultProbabilityOfHeadSideInB]
    iterationTimes
'''

def emMainprocess(samples, probability, mostIterations, threshold):
    it = 0
    while it < mostIterations:
        probabilityTmp = iterator(probability, samples)
        if numpy.abs(probability[0] - probabilityTmp[0]) < threshold and numpy.abs(probability[1] - probabilityTmp[1]) < threshold:
            break
        else:
            it += 1
            probability = probabilityTmp

    return [probabilityTmp, it]


'''
function declaration: the main process to solve the situationA with known classification
Parameters:
    samples - the sample set to be used
    labels  - the label of each sample set

Returns:
    [resultProbabilityOfHeadSideInA, resultProbabilityOfHeadSideInB]
'''

def solveSituationA(samples, labels):
    # to record the sum of each incident under two different (A & B)
    sums = [[0, 0], [0, 0]]
    for i in range(0, len(samples)):
        for sample in samples[i]:
            if sample == 0:
                sums[labels[i]][1] += 1
            elif sample == 1:
                sums[labels[i]][0] += 1
    thetaA = sums[0][0] / (sums[0][0] + sums[0][1])
    thetaB = sums[1][0] / (sums[1][0] + sums[1][1])
    return [thetaA, thetaB]


def main():
    # the sample input from the homeworkPDF for problem 1 & 2
    samples = numpy.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                           [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                           [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
    # the labels correspondent to the sample from the homeworkPDF for problem 1
    labels = numpy.array([1, 0, 0, 1, 0])

    choice = input("Please enter to choose to run:\n1 - situation A \n2 - situation B\ninput:")

    if choice == "2":
        results = emMainprocess(samples, [0.6, 0.5], 10000, 1e-18)
        print("Iteration times: ", results[1])
        print("Theta A: " + str(results[0][0]))
        print("Theta B: " + str(results[0][1]))
    elif choice == "1":
        results = solveSituationA(samples, labels)

        print("Theta A: " + str(results[0]))
        print("Theta B: " + str(results[1]))


if __name__ == '__main__':
    main()
