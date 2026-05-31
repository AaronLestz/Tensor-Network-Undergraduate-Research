# Tensor-Network-Undergraduate-Research
Repository containing my work for undergraduate research with professor Coley-O'Rourke

CURRENT: Don't have a provided task, but will try to generalize the 3 x 3 to n x n, as this is a reasonable step to also check if I need to change at all the setup around compressions. First thoughts is no, but regardless, it could also find some unexpected edge case.

FINISHED TASK: Get a working Hamiltonian contraction with Hamiltonian TN's as provided from the professor's code. Relevant committed files are:
- 3x3_hamiltonian_contractions.py
- wavefunction_generator.py (testing)

Other files meanings are mostly clear from their names. Only unclear one is the model_adapter.py, which is no longer in use, and was from the failed attempt at maintaining more of the framework seen in OUTDATED_3x3_hamiltonian_contractions.py. This failed due to being inherently incompatible (not enough/the right information) with the professor's code.