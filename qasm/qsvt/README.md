
## Quantum Singular Value Transformation (QSVT)

This is the collection of quantum circuits using Quantum Singular Value Transformation (QSVT) [1] for matrix inversion up to 32 qubits.

The matrix is choose to be a banded circulant matrix as following with explicit Block encoding given in [2] to prepare $A/4$. This matrix always has a condition number of $\kappa=\lambda_{\text{max}}/\lambda_{\text{min}}=5$ since its eigenvalues are exactly $\lambda_k=3-2\cos(2\pi k/N)$ where the matrix size is $N\times N$ and $k\in\{0,1,\cdots,N-1\}$. The overall circuit width is $\log_2(N)+4$ qubits with 3 ancillas for Block encoding purpose and 1 ancilla for implementing $R_z$ rotations for quantum signal processing.

```python 
alpha, beta, gamma  = 3, -1, -1

A = np.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
              [ beta, alpha, gamma,     0,     0,     0,     0,     0],
              [    0,  beta, alpha, gamma,     0,     0,     0,     0],
              [    0,     0,  beta, alpha, gamma,     0,     0,     0],
              [    0,     0,     0,  beta, alpha, gamma,     0,     0],
              [    0,     0,     0,     0,  beta, alpha, gamma,     0],
              [    0,     0,     0,     0,     0,  beta, alpha, gamma],
              [gamma,     0,     0,     0,     0,     0,  beta, alpha]])
```

The phase angles are taken from [this Pennylane dataset](https://pennylane.ai/datasets/inverse) with
```python 
import pennylane as qml

[dataset] = qml.data.load("other", name="inverse")
angles_qsvt = dataset.angles["qsvt"]["0.001"]["5"]   # matrix condition number kappa=5, target accuracy epsilon=0.001
phase_angles = qml.transform_angles(angles_qsvt, "QSVT", "QSP")  # convert to Wx convention and consistent with pyQSP
```


The phase angles can also be calculated from [pyqsp](https://github.com/ichuang/pyqsp) package for quantum singal processing without any small angle truncation and implemented in the circuit level via $R_z$ rotations. Notice some of the angles can be very small in the range of $<10^{-6}$ but are crucial for the accuracy. 
```python 
import pyqsp

kappa = 5 # matrix condition number
epsilon = 0.01 # target accuracy

pcoefs, scale = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
phase_angles = pyqsp.angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", chebyshev_basis=False, tolerance=1e-9)
```


Compilation detail:

```python 
transpiled_circuit = transpile(qsvt_circuit, basis_gates=['s', 'h', 'rz', 'x', 'z', 'cx'], optimization_level=0)
```


## Importance of QSVT
Quantum singular value transformation (QSVT) provides a unified framework to implement arbitrary polynomial functions of block-encoded operators. Given a block-encoding $U_A$ of a general matrix $A$, QSVT enables the transformation of the singular values of $A$ by a polynomial $f(\cdot)$. Many quantum algorithms have simple and near optimal implementations via the QSVT framework including Hamiltonian simulation [4], solving linear systems [2], Gibbs sampling [2], state preparation [5], topological data analysis [6], and amplitude amplification [7]. More specifically in this example, QSVT exhibits $O(\kappa \cdot \text{ploylog} (N) \log(\kappa/\epsilon))$ query complexity compare to the well-known HHL algorithm for solving linear system with $O(\kappa^2 \log(N)/\epsilon)$ complexity. More future work is needed to extend Block encoding to more types of matrix and handle larger matrix condition numbers [8].



## Reference 

[1] [Martyn, J. M., Rossi, Z. M., Tan, A. K., & Chuang, I. L. (2021). Grand unification of quantum algorithms. PRX quantum, 2(4), 040203.](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203) \
[2] [Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. (2019, June). Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. In Proceedings of the 51st annual ACM SIGACT symposium on theory of computing (pp. 193-204).](https://dl.acm.org/doi/pdf/10.1145/3313276.3316366) \
[3] [Camps, D., Lin, L., Van Beeumen, R., & Yang, C. (2024). Explicit quantum circuits for block encodings of certain sparse matrices. SIAM Journal on Matrix Analysis and Applications, 45(1), 801-827.](https://epubs.siam.org/doi/10.1137/22M1484298) \
[4] [Low, G. H., & Chuang, I. L. (2017). Optimal Hamiltonian simulation by quantum signal processing. Physical review letters, 118(1), 010501.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) \
[5] [O'Brien, O., & Sünderhauf, C. (2025). Quantum state preparation via piecewise QSVT. Quantum, 9, 1786.](https://quantum-journal.org/papers/q-2025-07-03-1786/) \
[6] [Berry, D. W., Su, Y., Gyurik, C., King, R., Basso, J., Barba, A. D. T., ... & Babbush, R. (2024). Analyzing prospects for quantum advantage in topological data analysis. PRX Quantum, 5(1), 010319.](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010319) \
[7] [Yoder, T. J., Low, G. H., & Chuang, I. L. (2014). Fixed-point quantum search with an optimal number of queries. Physical review letters, 113(21), 210501.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.210501) \
[8] [Novikau, I., & Joseph, I. (2025). Estimating QSVT angles for matrix inversion with large condition numbers. Journal of Computational Physics, 525, 113767.](https://www.sciencedirect.com/science/article/pii/S0021999125000506)