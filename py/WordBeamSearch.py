from __future__ import division
from __future__ import print_function

import numpy as np

from Beam import Beam, BeamList
from LanguageModel import LanguageModel


def wordBeamSearch(mat, beamWidth, lm, useNGrams):
    "decode matrix, use given beam width and language model"
    chars = lm.getAllChars()
    blankIdx = len(chars)  # blank label is supposed to be last label in RNN output
    maxT, _ = mat.shape  # shape of RNN output: TxC

    print("chars:", chars)
    print("blankIdx:", blankIdx)
    print("mat shape: ", mat.shape)

    genesisBeam = Beam(lm, useNGrams)  # empty string
    last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.addBeam(genesisBeam)  # start with genesis beam

    # go over all time-steps
    for t in range(maxT):
        # curr在当前时刻清零，旧的保存在last中
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  # get best beams
        for beam in bestBeams:
            # print("\n")
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.getText() != '':
                # char at time-step t must also occur at t-1
                labelIdx = chars.index(beam.getText()[-1])
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]
                # print("beam.getText: ", beam.getText())
                # print("labelIdx: ", labelIdx)
                # print("prNonBlank: %f=%f*%f"%(prNonBlank, beam.getPrNonBlank(), mat[t, labelIdx]))
            # else:
                # print("beam.getText2: ")

            # calc probability that beam ends with blank
            prBlank = beam.getPrTotal() * mat[t, blankIdx]
            # print("prBlank: %f=%f*%f"%(prBlank, beam.getPrTotal(), mat[t, blankIdx]))

            # save result
            curr.addBeam(beam.createChildBeam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.getNextChars()
            # print("nextChars:", nextChars, len(nextChars))
            for c in nextChars:
                # extend current beam with new character
                # print("\tcur char:", c)
                labelIdx = chars.index(c)
                if beam.getText() != '' and beam.getText()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                    # print("\tprNonBlank: %f=%f*%f"%(prNonBlank, beam.getPrBlank(), mat[t, labelIdx]))
                else:
                    prNonBlank = mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours
                    # print("\tprNonBlank2: %f=%f*%f"%(prNonBlank, beam.getPrTotal(), mat[t, labelIdx]))

                # save result
                curr.addBeam(beam.createChildBeam(c, 0, prNonBlank))

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.completeBeams(lm)
    bestBeams = last.getBestBeams(1)  # sort by probability
    return bestBeams[0].getText()


if __name__ == '__main__':
    testLM = LanguageModel('a b aa ab ba bb', 'ab ', 'ab')
    testMat = np.array([[0.3, 0.1, 0, 0.6], [0.3, 0.1, 0, 0.6]])
    # testMat = np.array([[0.1, 0.2, 0, 0.7], [0.15, 0.25, 0, 0.6]])
    testBW = 25
    res = wordBeamSearch(testMat, testBW, testLM, False)
    print('Result: "' + res + '"')
