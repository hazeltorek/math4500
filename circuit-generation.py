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

# crosstalking edge pairs, as given in paper
crosstalks = {
    "L6": {((0,1),(2,3)), ((1,2),(3,4)), ((2,3),(4,5))},
    "Y6": {((0,1),(2,3)), ((0,1),(2,5)), ((1,2),(3,4)), ((3,4),(2,5))},
    "G6": {((0,1),(3,4)), ((1,2),(4,5)), ((0,3),(1,4)), ((1,4),(2,5))},
}
 
# edge bidirectionality
for k, X in crosstalks.items():
    crosstalks[k] = X.union({((j,i),(k,l)) for ((i,j),(k,l)) in X})\
                     .union({((i,j),(l,k)) for ((i,j),(k,l)) in X})\
                     .union({((j,i),(l,k)) for ((i,j),(k,l)) in X})

# make sure edges are oriented bidirectionally 
for k in topologies.keys():
    # add the reverse orientation of each edge to the set as well
    topologies[k] = topologies[k].union(set((j, i) for (i, j) in list(topologies[k])))

# change for different runs
topology = topologies["L6"]
num_qubits = 6
# change accordingly for crosstalk dict
X = crosstalks["L6"]

# tolerance for sequential optimization 
epsilon = 1e-6

# assumed fidelity from paper
# beta = {(i, j): 0.9936 for (i, j) in topology}

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

# x_(q, i, j, t) = 1 iff qubit q is located at node i at time t and located at node j at time t+1
N = [{j for (i, j) in topology if i == n} for n in V]
x = m.binary_var_dict(((q, i, j, t) for t in T[:-1] for i in V for j in N[i].union({i}) for q in Q), name = "x")

m.add_constraints(w[q, i, t] == x[q, i, i, t] + m.sum(x[q, i, k, t] for k in N[i]) for q in Q for i in V for t in T[:-1])
m.add_constraints(w[q, i, t] == x[q, i, i, t-1] + m.sum(x[q, k, i, t-1] for k in N[i]) for q in Q for i in V for t in T[1:])

# m.add_constraints(x[p := G[t][0], i, i, t] + x[p, i, j, t] >= y[t][(p, G[t][1]), (i, j)] for t in T for (i, j) in topology)
# m.add_constraints(x[q := G[t][1], j, i, t] + x[q, j, j, t] >= y[t][(G[t][0], q), (i, j)] for t in T for (i, j) in topology)

#m.add_constraints(y[t][(G[t][0], G[t][1]), (i, j) >= 0] for t in T for (i, j) in topology)
#m.add_constraints(y[t][(p := G[t][0], q := G[t][1]), (i, j) >= w[p, i, t] + w[q, j, t] - 1] for t in T for (i, j) in topology)

# Gate McCormick Constraints
m.add_constraints(x[q, i, i, t*5] + x[q, i, j, t*5] >= y[t][q, (i, j)] for t in V[:-1] for (i, j) in topology for q in G[t])
m.add_constraints(w[q, i, t*5] >= y[t][q, (i, j)] for t in V[-1:] for (i, j) in topology for q in G[t])
m.add_constraints(y[t][G[t][0], (i, j)] >= w[G[t][0], i, t*5] + w[G[t][1], j, t*5] - 1 for t in V for (i, j) in topology)
m.add_constraints(y[t][G[t][1], (i, j)] >= w[G[t][1], i, t*5] + w[G[t][0], j, t*5] - 1 for t in V for (i, j) in topology)
# m.add_constraints(y[t][G[t][0], (i, j)] == y[t][G[t][1], (j, i)] for t in V for (i, j) in topology)
m.add_constraints(y[t][q, (i, j)] <= w[q, i, t*5] for t in V for (i, j) in topology for q in G[t])
m.add_constraints(y[t][G[t][0], (i, j)] == y[t][G[t][1], (j, i)] for t in V for (i, j) in topology)


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

# decision variable for crosstalk minimization
u = m.binary_var_dict(((t, e) for t in T for e in topology), name="u")
# constraints on cross-talking edges (ensuring u is correctly defined, from the paper)
# real timesteps — use y variables
m.add_constraints(u[t, (i,j)] == m.sum(y[t//5][q, (i,j)] for q in G[t//5]) for t in T if t % 5 == 0 for (i,j) in topology)
# dummy timesteps — use x variables (only defined for T[:-1])
m.add_constraints(u[t, (i,j)] == m.sum(x[q, i, j, t] for q in Q) for t in T[:-1] if t % 5 != 0 for (i,j) in topology)
# last timestep if it's dummy — x not defined so u must be 0
m.add_constraints(u[T[-1], (i,j)] == 0 for (i,j) in topology if T[-1] % 5 != 0)


# log prob of circuit success (maximizing, so optimize -o1)
#o1 = m.sum()

# circuit depth (minimizing dummy timesteps, minimize o2)
o2 = m.sum(z[t] for t in D)

# crosstalk (not sure if we will impl this one)
o3 = m.sum(u[t, e1] * u[t, e2] for t in T for (e1, e2) in X)

# Sequential Optimization
m.minimize(o2)
s = m.solve()
# m.add_constraint(o2 <= o2val + epsilon)
# m.minimize(o3)
# m.print_solution()

# generate qubit swaps based on timestep and qubit pair from solutions to x
swaps = [(q, i, j, t) for (q, i, j, t) in x if s[f"x_{q}_{i}_{j}_{t}"] == 1]

# build new circuit by inserting the required gates at each timestep
qc = QuantumCircuit(num_qubits)
for t in T:
    if (t % 5 != 0): # dummy timestep
        # collect any swaps that need to happen at this timestep
        curr_swaps = list(filter(lambda s: t == s[3] and s[1] != s[2], swaps))
        assert(len(curr_swaps) <= 1)
    else: # real timestep  
        # copy the data from the original circuit
        g1 = circuit.data[t // 5]
        g2 = g1.replace(qubits = tuple(qc.qubits[circuit.find_bit(q)[0]] for q in g1.qubits))
        qc.append(g2)

print(circuit)
print(qc)

"""
m.minimize(-o1)
m.solve()
o1_val = m.objective_value
m.add_constraint(-o1 <= o1_val + epsilon, ctname = "ensure_error_rate_optimality")
m.minimize(o2)
m.solve() 
"""