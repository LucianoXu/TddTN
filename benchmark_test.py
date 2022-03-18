
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

import tdd_origin

from qiskit_phaser import SimQiskitCir, SimQiskitCir_tdd

def timing(method, count=1):
    t1 = time.perf_counter()
    for i in range(count):
        method()
    t2 = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(t2-t1, (t2-t1)/count))
    return (t2-t1)/count


pytdd.setting_update(4, False, True, 3e-7, 0.3, 10000)
path="Benchmarks/"
file_name="quantum_volume_n10_d5.qasm"

do_numpy_backend = False

def PytorchCalc():
    global cir, U_old
    U_old = SimQiskitCir(cir)

def PytddCalc():
    global cir, U_new
    U_new = SimQiskitCir_tdd(cir)

for m in range(1):

    if do_numpy_backend:
        print("=====================================================\n")
        tn.set_default_backend('numpy')
        cir=QuantumCircuit.from_qasm_file(path+file_name)
        timing(PytorchCalc)
        print("\n")



        # convert to Xin Hong's TDD
        s = U_old.shape

        tdd_origin.TDD.Ini_TDD([str(i) for i in range(len(s))])
        var=[]
        for i in range(len(s)//2):
            var.append(tdd_origin.TN.Index(str(i)))
            var.append(tdd_origin.TN.Index(str(i + len(s)//2)))

        ts1=tdd_origin.TN.Tensor(U_old,var)
        ts1.tdd().show()
        ##############

    print("=====================================================\n")
    tn.set_default_backend('pytdd')
    pytdd.clear_cache()
    
    cir=QuantumCircuit.from_qasm_file(path+file_name)
    timing(PytddCalc)
    print()
    tdd_size = U_new.size()
    print("tdd result size: ", tdd_size)
    #U_new.tensor.show(str(m))
    print()
    print("tdd info:")

    print(U_new.info)
    #U_new.show()


    #print(str(U_new))
    '''print("\n")
    print("\n\n\n\n\n######################################################")
    print(U_old-U_new.val)
    '''
    if do_numpy_backend:
        print("max diff: ",np.max(U_old - U_new.numpy()))


#pytdd.clear_cache()
pytdd.test()