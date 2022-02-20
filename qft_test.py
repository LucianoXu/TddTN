
import numpy as np
import tensornetwork as tn
import time
import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from func_timeout import func_set_timeout
import pandas as pd
import random
import sys

def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]
        
def apply_iostate(qubit_edges,states):
    for k in range(len(qubit_edges)):
        if states[k]==0:
            op = tn.Node(np.array([1,0]))
        else:
            op = tn.Node(np.array([0,1]))
        tn.connect(qubit_edges[k], op[0])

def matrix_2_np(U,q):
    if len(q)!=2:
        return U
    U_new=np.zeros((2, 2, 2, 2), dtype=complex)
    for k1 in range(2):
        for k2 in range(2):
            for k3 in range(2):
                for k4 in range(2):
                    U_new[k1][k2][k3][k4]=U[2*k2+k1][2*k4+k3]
    return U_new

def np_2_matrix(U,q_num):
    U_new=np.zeros((2**q_num,2**q_num), dtype=complex)
    for k1 in range(2**q_num):
        for k2 in range(2**q_num):
            b1=[int(b1) for b1 in list(bin(k1)[2:])]
            for k3 in range(q_num-len(b1)):
                b1.insert(0,0)
            b2=[int(b2) for b2 in list(bin(k2)[2:])]
            for k3 in range(q_num-len(b2)):
                b2.insert(0,0)            
            temp=U
            for b in b1:
                temp=temp[b]
            for b in b2:
                temp=temp[b]
            U_new[k1][k2]=temp
    return U_new

def get_real_qubits_num(cir):
    gates=cir.data
    q=0
    for k in range(len(gates)):
        q=max(q,max([qbit.index for qbit in gates[k][1]]))
    return q+1


@func_set_timeout(3600)
def Simulation_with_TensorNetwork(cir,io=False):
    all_nodes = []
    with tn.NodeCollection(all_nodes):
        qubits_num=get_real_qubits_num(cir)
        state_nodes = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(qubits_num)
        ]
        
        qubits0=[node[0] for node in state_nodes]
        qubits = [node[1] for node in state_nodes]
        gates=cir.data
        for k in range(len(gates)):
            U=Operator(gates[k][0]).data
            q=[qbit.index for qbit in gates[k][1]]
            if len(q)==2:
                U=matrix_2_np(U,q)
            apply_gate(qubits, U, q)
    
    
        if io:
            input_state=[random.randint(0,1) for k in range(qubits_num)]
            output_state=[random.randint(0,1) for k in range(qubits_num)]
            
            apply_iostate(qubits0,input_state)
            apply_iostate(qubits,output_state)
            qubits0=[]
            qubits=[]
        edge_order=qubits0+qubits
#         print(edge_order)
        result = tn.contractors.greedy(all_nodes, output_edge_order=edge_order).tensor
        return result

def Simulation_with_time_out(cir,io=False):
    try:
        return Simulation_with_TensorNetwork(cir,io)
    except:
        return 0






path="Benchmarks/"
file_name="qft_10.qasm"


print("=====================================================\n")
tn.set_default_backend('numpy')
cir=QuantumCircuit.from_qasm_file(path+file_name)
t_start= time.time()
U_old=Simulation_with_TensorNetwork(cir)
t_end=time.time()
print('Time:',t_end-t_start)
print("\n")




#print(str(U_old))



print("=====================================================\n")
tn.set_default_backend('tdd')
cir=QuantumCircuit.from_qasm_file(path+file_name)
t_start= time.time()
U_new=Simulation_with_TensorNetwork(cir)
t_end=time.time()
print('Time:',t_end-t_start)
#U.show()
print("\n")
print(U_new.size())
print(U_new.info)



#print(str(U_new))
'''print("\n")
print("\n\n\n\n\n######################################################")
print(U_old-U_new.val)
'''
print("max diff: ",np.max(U_old - U_new.numpy()))
