import numpy as np

# pythran export testFunc(float[][])
def testFunc(signal):
    A = np.fft.rfft(signal)
    return A
