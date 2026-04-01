from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume

# create hardware topology just as a set of edges
top_L6 = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}
top_Y6 = {(1, 2), (2, 3), (3, 6), (3, 4), (4, 5)}
top_G6 = {(1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (2, 5), (3, 6)}
top_L8 = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)}
top_Y8 = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (7, 8)}
top_G8 = {(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (1, 5), (2, 6), (3, 7), (4, 8)}

# change for different runs
topology = top_L6
num_qubits = 6

# generate a 6-qubit example circuit
circuit = quantum_volume(num_qubits = num_qubits, depth = num_qubits)

# https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/tutorials/Linear_Programming.ipynb
from docplex.mp.model import Model
from functools import reduce
import itertools as it

m = Model(name = "circuit-optimizer")

# setting up decision variables described in 3.1 > Location Variables and Constraints
Q = range(num_qubits + 1)      # 6 qubits in the example circuit
V = range(num_qubits + 1)      # 6 qubits in the hardware topology
T = range(num_qubits * 5 + 1)  # 4 dummy and 1 real timestep for each depth, plus one for exclusive index

# https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html#docplex.mp.model.Model.binary_var
w = m.binary_var_cube(Q, V, T, name = "w")

# setting up decision variables described in 3.1 > Gate Variables and Constraints
# need to define this one separately for each timestep t since the set of gates is indexed by t
G = list(it.chain(*(list(it.repeat((), 4)) + [g] for g in [tuple(circuit.find_bit(q).index for q in circuit.data[t].qubits) for t in range(num_qubits + 1)])))
H = list(it.product(range(num_qubits + 1), range(num_qubits + 1)))
y = m.binary_var_cube(G, H, T, name = "y")

# constraint: each gate is impl. once
for (t, g) in it.product(T, G):
    if g is (): continue
    m.add_constraint(reduce(lambda x, y: x + y, [y[g][h][t] for h in topology]))