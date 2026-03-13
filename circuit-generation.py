# generate a 6-qubit example circuit
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume

circuit = quantum_volume(num_qubits =  6, depth = 6)
circuit.draw(output = "mpl")

# figure out the CPLEX stuff
# https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/tutorials/Linear_Programming.ipynb
from docplex.mp.model import Model
import itertools

m = Model(name = "circuit-optimizer")

# setting up decision variables described in 3.1 > Location Variables and Constraints
Q = range(7)            # 6 qubits in the example circuit
V = range(7)            # 6 qubits in the hardware topology
T = range(6 * 5 + 1)    # 4 dummy and 1 real timestep for each depth, plus one for exclusive index

# https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html#docplex.mp.model.Model.binary_var
w = m.binary_var_cube(Q, V, T, name = "w")

# setting up decision variables described in 3.1 > Gate Variables and Constraints
# need to define this one separately for each timestep t since the set of gates is indexed by t
y = m.binary_var_matrix(name = "y")