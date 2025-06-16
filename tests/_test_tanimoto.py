import unittest
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from tanimoto import tanimoto_similarity


def smiles_to_morgan_fingerprint(smile, radius=2, n_bits=2048):
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(n_bits)  # Return a zero vector for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return fp


def ground_truth(A: list, B: list):
    N = len(A)
    M = len(B)
    rdkit_similarities = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            rdkit_similarities[i, j] = DataStructs.TanimotoSimilarity(
                A[i], B[j]
            )

    return rdkit_similarities

class TestTensorizedTanimoto(unittest.TestCase):
    # 10
    simple = [
        "CCO",  # Ethanol
        "CCC",  # Propane
        "CCN",  # Ethylamine
        "CCOCC",  # Diethyl ether
        "CC(=O)O",  # Acetate
        "CCOCCO",  # Ethylene glycol diethyl ether
        "CC(C)O",  # Isopropanol
        "C1CCCCC1",  # Cyclohexane
        "C1=CC=CC=C1",  # Benzene
        "CC(C)C",  # Isobutane
    ]

    # 8
    pharma = [        
        "CC(C)Cc1ccc(cc1)O",  # Ibuprofen (Painkiller, NSAID)
        "COc1ccc(cc1)C(O)CN(C)C",  # Paracetamol (Painkiller, Fever reducer)
        "CCN1C(=O)c2c(ncn2C)n(c1=O)C",  # Caffeine (Stimulant)
        "CCOC(=O)c1c(cc(cc1O)O)C(=O)O",  # Aspirin (Painkiller, NSAID)
        "CN(C)C(=O)c1ccc(cc1)O",  # Acetaminophen (Analgesic)
        "COC(=O)c1c(nc(nc1O)N)N",  # Methotrexate (Chemotherapy, Autoimmune diseases)
        "CN(C)C(=O)c1c(cc(cc1O)O)O",  # Dopamine (Neurotransmitter, Parkinson's treatment)
        "O=C(O)c1ccccc1O",  # Salicylic Acid (Pain relief, Skincare)
    ]

    def _compute_fps(self):
        self.rdkit_simple = [smiles_to_morgan_fingerprint(s) for s in self.simple]
        self.rdkit_pharma = [smiles_to_morgan_fingerprint(s) for s in self.pharma]
        self.rdkit = self.rdkit_pharma + self.rdkit_pharma

        self.np_simple = np.array([list(fp) for fp in self.rdkit_simple])
        self.np_pharma = np.array([list(fp) for fp in self.rdkit_pharma])
        self.np = np.array([list(fp) for fp in self.rdkit])

        self.pt_simple = torch.tensor(self.np_simple, dtype=torch.float32)
        self.pt_pharma = torch.tensor(self.np_pharma, dtype=torch.float32)
        self.pt = torch.tensor(self.np, dtype=torch.float32)

    def setUp(self):
        self._compute_fps()

        self.gt_all = ground_truth(self.rdkit, self.rdkit)  # 18x18
        self.gt_simple_pharma = ground_truth(self.rdkit_simple, self.rdkit_pharma)  # 10x8
        self.gt_simple = ground_truth(self.rdkit_simple, self.rdkit_simple) # 10x10
        self.gt_pharma = ground_truth(self.rdkit_pharma, self.rdkit_pharma) # 8x8

        self.inputs = [
            (self.rdkit, self.rdkit),
            (self.rdkit_simple, self.rdkit_pharma),
            (self.rdkit_simple, self.rdkit_simple),
            (self.rdkit_pharma, self.rdkit_pharma)
        ]

    def test_tanimoto_similarity(self):
        # Compute pairwise similarity using tensorized function

        for method in ["rdkit", "torch", "numpy", "scipy"]:
            for inputs in self.inputs:
                expected = ground_truth(*inputs)
                output = tanimoto_similarity(*inputs, method=method)

                with self.subTest(inputs=inputs, method=method):
                    # Check consistency
                    np.testing.assert_allclose(
                        output, expected, rtol=1e-6, atol=1e-6
                    )
        


if __name__ == "__main__":
    unittest.main()