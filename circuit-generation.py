from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume

from docplex.mp.model import Model
from functools import reduce
import itertools as it

# create hardware topology just as a set of edges
topologies = {
    "L6": {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)},
    "Y6": {(1, 2), (2, 3), (3, 6), (3, 4), (4, 5)},
    "G6": {(1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (2, 5), (3, 6)},
    "L8": {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)},
    "Y8": {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (7, 8)},
    "G8": {(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (1, 5), (2, 6), (3, 7), (4, 8)}
}

# make sure edges are oriented bidirectionally 
for k in topologies.keys():
    # add the reverse orientation of each edge to the set as well
    topologies[k] = topologies[k].union(set((j, i) for (i, j) in list(topologies[k])))

# change for different runs
topology = topologies["L6"]
num_qubits = 6

# generate a 6-qubit example circuit
circuit = quantum_volume(num_qubits = num_qubits, depth = num_qubits)

# https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/tutorials/Linear_Programming.ipynb
m = Model(name = "circuit-optimizer")

# setting up decision variables described in 3.1 > Location Variables and Constraints
Q = range(num_qubits)      # 6 qubits in the example circuit
V = range(num_qubits)      # 6 qubits in the hardware topology
T = range(num_qubits * 5)  # 4 dummy and 1 real timestep for each depth

# https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html#docplex.mp.model.Model.binary_var
# w_(q, i, t) = 1 iff qubit q resides at node i (in hardware) at time t, described under 3.1 > Location
w = m.binary_var_cube(Q, V, T, name = "w")
m.add_constraints(m.sum(w[q, v, t] for v in V) == 1 for (q, t) in it.product(Q, T)) # each qubit is at exactly one node
m.add_constraints(m.sum(w[q, v, t] for q in Q) == 1 for (v, t) in it.product(V, T)) # each node hosts exactly one qubit

# y_((p,q), (i, j), t) = 1 iff gate (p, q) is impl on hardware arc (i, j) at time t
G = [tuple(circuit.find_bit(q).index for q in circuit.data[t].qubits) for t in V] # logical gates at timestep t
y = [m.binary_var_matrix(G[t], topology, name = f"y{t}") for t in V]

# TODO: still need to add constraints on y, this is the bit with mccormick constraints

# x_(q, i, j, t) = 1 iff qubit q is located at node i at time t and located at node j at time t+1
N = [{(i, j) for (i, j) in topology if i == n} for n in V]
x = m.binary_var_dict((q, i, j, t) for t in T[:-1] for i in V for j in N[i] for q in Q)

# TODO: still need to add constraints on x, see page 7  
