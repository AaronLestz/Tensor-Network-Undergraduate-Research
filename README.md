# Tensor-Network-Undergraduate-Research
Repository containing subset of early work for undergraduate research, with guidance for projects to complete being provided by Professor Coley-O'Rourke.

Last update to any of the code: August 2026

MOST RECENT FINISHED TASK: Get a working Hamiltonian contraction with Hamiltonian TN's as provided from the professor's code. Relevant committed files are:
- 3x3_hamiltonian_contractions.py
- wavefunction_generator.py (testing)

Along with the tools required for exact 2-D and 3-D tensor network contraction - Linear
Algebra, NumPy, and Algorithms - the contractions in this repository also implement:
Compression, Canonical Forms of Matrix Product States, Error Calculation, and
Normalization & Log Scaling of results

With the exception of OUTDATED_3x3_hamiltonian_contractions.py, all tensor network
contraction algorithms in this repository are correctly implemented. Additionally, 
with the exception of 3x3_hamiltonian_contractions.py (and its outdated version), all contraction files work by themselves - i.e., they do not require being in the same directory any other file.

Both of the hamiltonian contraction algorithms rely on files provided to me by the
professor, which are *not* contained within the repository, and so they will not be
able to run with only the code in the repository. However, understanding of most of each
algorithm's code does not rely on said files, and thus behavior of each algorithm should
still be largely evident from this repository alone. 