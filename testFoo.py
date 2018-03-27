import Foo
import numpy as np
from BaseClasses.Utils import *

A = np.stack((np.linspace(1,2,64),np.linspace(2,3,64)),axis=1)
#A = np.linspace(1,2,64)
B = Foo.testFunc(A)
print len(B)
print B
C = np.fft.rfft(A)
print "expected"
print C
# printList(B)
# printList(C)
# printList(B-C)
print np.max(np.abs(B-C))

