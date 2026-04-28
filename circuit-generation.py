from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume

from docplex.mp.model import Model
from functools import reduce
import itertools as it

# create hardware topology just as a set of edges
topologies = {
    "L6": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)},
    "Y6": {(0, 1), (1, 2), (2, 5), (2, 3), (3, 4)},
    "G6": {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)},
    "L8": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)},
    "Y8": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)},
    "G8": {(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)}
}

# make sure edges are oriented bidirectionally 
for k in topologies.keys():
    # add the reverse orientation of each edge to the set as well
    topologies[k] = topologies[k].union(set((j, i) for (i, j) in list(topologies[k])))

# change for different runs
topology = topologies["L6"]
num_qubits = 6

# assumed fidelity from paper
beta = {(i, j): 0.9936 for (i, j) in topology}

# epsilon tolerance for CPLEX to work properly with integer values (fp errors can exist)
epsilon = 1e-6

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

# each gate is impl. only exactly once at any given timestep
m.add_constraints(m.sum(y[t][G[t][0], arc] for arc in topology) == 1 for t in V)
m.add_constraints(m.sum(y[t][G[t][1], arc] for arc in topology) == 1 for t in V)

# x_(q, i, j, t) = 1 iff qubit q is located at node i at time t and located at node j at time t+1
N = [{j for (i, j) in topology if i == n} for n in V]
x = m.binary_var_dict((q, i, j, t) for t in T[:-1] for i in V for j in N[i].union({i}) for q in Q)

# McCormick Constraints from the paper. I don't really know if possible to do in a nice, one-liner :(
for t in V:
    p, q = G[t]
    for (i, j) in topology:
        yp = y[t][p, (i, j)]  # p assigned to node i on arc (i,j)
        yq = y[t][q, (i, j)]  # q assigned to node i on arc (i,j)

        # Basic lower bounds for both
        m.add_constraint(yp >= 0)
        m.add_constraint(yq >= 0)

        if t < V[-1]: # use x variables for stronger upper bounds
            m.add_constraint(yp <= x[p, i, i, t] + x[p, i, j, t]) 
            m.add_constraint(yq <= x[q, i, i, t] + x[q, i, j, t])
            m.add_constraint(yp >= w[p, i, t] + w[q, j, t] - 1)
            m.add_constraint(yq >= w[q, i, t] + w[p, j, t] - 1)
        else:         # fallback McCormick constraints for last timestep
            m.add_constraint(yp <= w[p, i, t])
            m.add_constraint(yp >= w[p, i, t] + w[q, j, t] - 1)
            m.add_constraint(yq <= w[q, i, t])
            m.add_constraint(yq >= w[q, i, t] + w[p, j, t] - 1)

# TODO: missing the mcormick bit at the bottom of page 7
m.add_constraints(w[q, i, t] == x[q, i, i, t] + m.sum(x[q, i, k, t] for k in N[i]) for q in Q for i in V for t in T[:-1])
m.add_constraints(w[q, i, t] == x[q, i, i, t-1] + m.sum(x[q, k, i, t-1] for k in N[i]) for q in Q for i in V for t in T[1:])

# the paper has the index of t as T[:-1] here, but i think we can restrict to V[:-1] since G is empty on dummy timesteps
# TODO: do i need to multiply the indices to account for dummy timesteps
m.add_constraints(x[G[t][0], i, j, t] == x[G[t][1], j, i, t] for t in V[:-1] for (i, j) in topology)

# i believe there is a typo in the paper here: sum index should be G^t not Q^t
# the paper has the index of t as T[:-1] here, but i think we can restrict to V[:-1] since G is empty on dummy timesteps
m.add_constraints(m.sum(x[q, i, j, t] for q in set(Q).difference(set(G[t]))) == m.sum(x[p, j, i, t] for p in set(Q).difference(set(G[t]))) for t in V[:-1] for (i, j) in topology)

# dummy timesteps (timesteps with no gates)
D = [t for t in T[:-1] if (t % 5 != 0) or not G[t // 5]]

#print(x.keys())
# z^t = 1 iff qubit x_(q, i, j, t) = 1 (moves at dummy timestep t)
z = m.binary_var_dict(D, name = "z")
m.add_constraints(m.sum(x[q, i, j, t] for (i, j) in topology) <= z[t] for t in D for q in Q) # dummy timesteps only 1 when qubit moves
m.add_constraints(z[D[t]] >= z[D[t+1]] for t in range(len(D)-1))
# ^ above is symmetry breaking for CPLEX speed (only consider doing dummy timesteps in one order) 
# can be removed without breaking program if not working

# log prob of circuit success (maximizing, so optimize -o1)
#o1 = m.sum()

# circuit depth (minimizing dummy timesteps, minimize o2)
o2 = m.sum(z[t] for t in D)

# crosstalk (not sure if we will impl this one)
# o3 = m.sum()

# Sequential Optimization
m.minimize(o2)
m.solve() 
m.print_solution()

"""
m.minimize(-o1)
m.solve()
o1_val = m.objective_value
m.add_constraint(-o1 <= o1_val + epsilon, ctname = "ensure_error_rate_optimality")
m.minimize(o2)
m.solve() 
"""