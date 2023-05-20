
import numpy as np
import tensornetwork as tn
import tddpy
import time
import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
# from func_timeout import func_set_timeout
# import pandas as pd
import random
import sys

import tdd_origin

from tnparser import qiskit_parser

def timing(method, count=1):
    t1 = time.perf_counter()
    for i in range(count):
        method()
    t2 = time.perf_counter()
    print('total time: {}s, average time: {}s'.format(t2-t1, (t2-t1)/count))
    return (t2-t1)/count


tddpy.reset(4, False, True, 3e-7, 0.5, 13000)
tddpy.TDD.check_parameter(False)

path="Benchmarks/"
file_name="mod8-10_178_squeezed.qasm"

do_numpy_backend = True

def PytorchCalc():
    global cir, U_matrix
    U_matrix = qiskit_phaser.SimQiskitCir(cir)

def TddPyCalc():
    global cir, U_tddpy
    U_tddpy = qiskit_phaser.SimQiskitCir_tdd(cir)

for m in range(1):

    if do_numpy_backend:
        print("=====================================================\n")
        tn.set_default_backend('CUDAcpl')
        cir=QuantumCircuit.from_qasm_file(path+file_name)
        timing(PytorchCalc)
        U_matrix = tddpy.CUDAcpl.CUDAcpl2np(U_matrix.tensor)
        print("\n")

        

        
        # convert to Xin Hong's TDD
        s = U_matrix.shape

        tdd_origin.TDD.Ini_TDD([str(i) for i in range(len(s))])
        var=[]
        
        for i in range(len(s)//2):
            var.append(tdd_origin.TN.Index(str(i)))
            var.append(tdd_origin.TN.Index(str(i + len(s)//2)))
        
        ts1=tdd_origin.TN.Tensor(U_matrix,var)
        ori_tdd = ts1.tdd()
        
        print("Original TDD size: ",ori_tdd.size())

        ori_tdd.show()
        ##############

        # convert into TddPy
        storage_order = []
        for i in range(len(s)//2):
            storage_order.append(i)
            storage_order.append(i + len(s)//2)

        U_tddpy_converted = tddpy.TDD.as_tensor((U_matrix, 0, storage_order))
        print("TddPy (converted from matrix) size: ", U_tddpy_converted.size())
        #################

    print("=====================================================\n")
    tn.set_default_backend('tddpy')
    tddpy.clear_cache()
    
    cir=QuantumCircuit.from_qasm_file(path+file_name)
    timing(TddPyCalc)
    print()
    tdd_size = U_tddpy.size()
    print("tddpy result size: ", tdd_size)
    U_tddpy.tensor.show("tddpy")
    print()
    print("tdd info:")

    print(U_tddpy.info)


    if do_numpy_backend:
        pass
        print("max diff: ",np.max(U_matrix - U_tddpy.numpy()))
    


    #tddpy.clear_cache()
    #tddpy.test()