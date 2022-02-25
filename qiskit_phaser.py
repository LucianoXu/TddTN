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
    with tn.NodeCollection(all_nodes):
        qubit_num = len(cir.qubits)
        state_nodes = [
            tn.Node(np.eye(2, dtype=complex)) for _ in range(qubit_num)
        ]
        qubit_edges_L=[node[0] for node in state_nodes]
        qubit_edges_R = [node[1] for node in state_nodes]
        gates=cir.data
        for gate in gates:
            U=Operator(gate[0]).data.reshape((2,)*2*len(gate[1]))
            q=[qbit.index for qbit in gate[1]]
            apply_gate(qubit_edges_R, U, q)

        edge_order = qubit_edges_L + qubit_edges_R

        result = tn.contractors.auto(all_nodes, output_edge_order=edge_order).tensor
        return result
