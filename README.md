# MATH 4500 Final Project: Qubit Assignment and Routing via Integer Programming

This is an implementation of the mathematical model for qubit routing provided in Qubit Assignment and Routing via Integer Programming by Nannicini et al., which was published in *ACM Transactions on Quantum Computing* in October 2022.

### Usage

This assumes you have a working install of Python version 3.14 and the academic version of CPLEX with version 22.1.2.0. To install the required Python library dependencies:

```bash
pip install -r requirements.txt
```

Then, to run the model:

```bash
python circuit-generation.py
python analyze.py
```

Depending on whether you are performing sequential optimization or just focusing on one of the specific objective functions, you may need to comment or uncomment the code at the bottom of `./circuit-generation.py`.