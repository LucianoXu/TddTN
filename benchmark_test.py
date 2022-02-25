
import numpy as np
import tensornetwork as tn
import pytdd
import time
import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from func_timeout import func_set_timeout
import pandas as pd
import random
import sys

from qiskit_phaser import SimQiskitCir



path="Benchmarks/"
file_name="qft_10.qasm"

do_numpy_backend = True

for i in range(3):

    if do_numpy_backend:
        print("=====================================================\n")
        tn.set_default_backend('numpy')
        cir=QuantumCircuit.from_qasm_file(path+file_name)
        t_start= time.time()
        U_old=SimQiskitCir(cir)
        t_end=time.time()
        print('Time:',t_end-t_start)
        print("\n")




    print("=====================================================\n")
    tn.set_default_backend('pytdd')
    
    cir=QuantumCircuit.from_qasm_file(path+file_name)
    t_start= time.time()
    U_new = SimQiskitCir(cir)
    t_end=time.time()
    print()
    print('Time:',t_end-t_start)
    #U.show()
    print()
    print("tdd result size: ", U_new.size())
    print()
    print("tdd info:")

    print(U_new.info)


    #print(str(U_new))
    '''print("\n")
    print("\n\n\n\n\n######################################################")
    print(U_old-U_new.val)
    '''
    if do_numpy_backend:
        print("max diff: ",np.max(U_old - U_new.numpy()))
