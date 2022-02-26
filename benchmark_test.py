
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

from qiskit_phaser import SimQiskitCir, SimQiskitCir_tdd

def timing(method, count=1):
    t1 = time.perf_counter()
    for i in range(count):
        method()
    t2 = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(t2-t1, (t2-t1)/count))
    return (t2-t1)/count



path="Benchmarks/"
file_name="quantum_volume_n8_d5.qasm"

do_numpy_backend = True

def PytorchCalc():
    global cir, U_old
    U_old = SimQiskitCir(cir)

def PytddCalc():
    global cir, U_new
    U_new = SimQiskitCir_tdd(cir)

for i in range(3):

    if do_numpy_backend:
        print("=====================================================\n")
        tn.set_default_backend('numpy')
        cir=QuantumCircuit.from_qasm_file(path+file_name)
        timing(PytorchCalc)
        print("\n")




    print("=====================================================\n")
    tn.set_default_backend('pytdd')
    
    cir=QuantumCircuit.from_qasm_file(path+file_name)
    timing(PytddCalc)
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
