import numpy as np


class PairingEnergies:
    def __init__(self, alphabet_list=("A", "C", "G", "U"), energy_matrix: np.array = None):
        self.alphabet = alphabet_list
        self.alphabet_size = len(self.alphabet)
        self.energy_matrix = None
        self._init_matrix(energy_matrix, rand_init=False)

    def _init_matrix(self, energy_matrix: np.array, rand_init=False):
        if energy_matrix is not None:
            assert len(energy_matrix) == self.alphabet_size
            self.energy_matrix = energy_matrix
        else:
            self.energy_matrix = np.zeros((self.alphabet_size, self.alphabet_size))
            if rand_init:
                self._init_matrix_random()

    def _init_matrix_random(self):
        for i, base1 in enumerate(self.alphabet):
            for j, base2 in enumerate(self.alphabet):
                if (base1, base2) in {('G', 'C'), ('C', 'G'), ('A', 'U'), ('U', 'A'), ('G', 'U'), ('U', 'G')}:
                    self.energy_matrix[i, j] = np.random.uniform(-3.0, -1.0)
                else:
                    self.energy_matrix[i, j] = np.random.uniform(0.0, 3.0)

    def _idx(self, letter: str) -> int:
        return self.alphabet.index(letter)

    def set_noncanonical(self, energy):
        for i, base1 in enumerate(self.alphabet):
            for j, base2 in enumerate(self.alphabet):
                if (base1, base2) not in {('G', 'C'), ('C', 'G'), ('A', 'U'), ('U', 'A'), ('G', 'U'), ('U', 'G')}:
                    self.energy_matrix[i, j] = energy

    def set(self, l1: str, l2: str, energy):
        assert l1 in self.alphabet and l2 in self.alphabet
        i1 = self._idx(l1)
        i2 = self._idx(l2)
        bf = energy
        self.energy_matrix[i1][i2] = bf
        self.energy_matrix[i2][i1] = bf

    def paired(self, l1: str, l2: str) -> float:
        return self.energy_matrix[self._idx(l1), self._idx(l2)]

    def unpaired(self, l="") -> float:
        return 0.0

    def get_energy_mat(self):
        return self.energy_matrix

    def __str__(self) -> str:
        # print("-- Boltzmann factors:")
        # print(self.alphabet_list)
        # print(self.energy_matrix)
        return f"-- Energies:\n{self.alphabet}\n{self.energy_matrix}"
