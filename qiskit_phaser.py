import tensornetwork as tn
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
import numpy as np

def apply_gate(qubit_edges, gate, operating_qubits):
    op = tn.Node(gate)

    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]


def SimQiskitCir(cir):
    '''
        Take in the circuit in qiskit.QuantumCircuit, and return the corresponding result.
    '''
    all_nodes = []
    qubit_num = len(cir.qubits)
    gates=cir.data

    with tn.NodeCollection(all_nodes):

        state_nodes = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(qubit_num)
        ]

        qubit_edges_L=[node[0] for node in state_nodes]
        qubit_edges_R = [node[1] for node in state_nodes]
        for gate in gates:
            U=Operator(gate[0]).data.reshape((2,)*2*len(gate[1]))
            q=[qbit.index for qbit in gate[1]]
            apply_gate(qubit_edges_R, U, q)

        edge_order = qubit_edges_L + qubit_edges_R

    result = tn.contractors.auto(all_nodes, output_edge_order=edge_order).tensor
    return result





def apply_gate_tdd(qubit_edges, gate, operating_qubits, order):
    op = tn.Node((gate, order))

    for i, bit in enumerate(operating_qubits):
        tn.connect(qubit_edges[bit], op[i])
        qubit_edges[bit] = op[i + len(operating_qubits)]

def get_order(qubit_current_index, operating_qubits):
    order_l = [0]*len(operating_qubits)
    order_r = [0]*len(operating_qubits)
    for i, bit in enumerate(operating_qubits):
        order_l[i] = qubit_current_index[bit]
        qubit_current_index[bit] += 1
        order_r[i] = qubit_current_index[bit]

    return order_l + order_r

def SimQiskitCir_tdd(cir):
    '''
        Take in the circuit in qiskit.QuantumCircuit, and return the corresponding result.
    '''
    all_nodes = []
    qubit_num = len(cir.qubits)
    gates=cir.data


    with tn.NodeCollection(all_nodes):
        # first arrange the global order

        # count how many indices are needed on each qubit. 
        # starting in 2 because we additionally add the identity matrix in front of the whole circuit
        qubit_edge_numbers = [1] * qubit_num
        for gate in gates:
            q=[qbit.index for qbit in gate[1]]
            for i,bit in enumerate(q):
                qubit_edge_numbers[bit] += 1

        qubit_current_index = [0] * qubit_num
        for i in range(1, qubit_num):
            qubit_current_index[i] = qubit_current_index[i-1] + 1 + qubit_edge_numbers[i]


        #first 
        state_nodes = [
            tn.Node((np.eye(2, dtype=complex),[qubit_current_index[i],qubit_current_index[i]+1])) for i in range(qubit_num)
        ]
        for i in range(qubit_num):
            qubit_current_index[i] += 1

        qubit_edges_L=[node[0] for node in state_nodes]
        qubit_edges_R = [node[1] for node in state_nodes]


        for gate in gates:
            U=Operator(gate[0]).data.reshape((2,)*2*len(gate[1]))
            q=[qbit.index for qbit in gate[1]]
            apply_gate_tdd(qubit_edges_R, U, q, get_order(qubit_current_index, q))

        edge_order = qubit_edges_L + qubit_edges_R

    result = tn.contractors.auto(all_nodes, output_edge_order=edge_order).tensor
    return result
