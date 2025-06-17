import pytest
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

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


@pytest.fixture
def simple_smiles():
    """Simple molecules for testing."""
    return [
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


@pytest.fixture
def pharma_smiles():
    """Pharmaceutical molecules for testing."""
    return [        
        "CC(C)Cc1ccc(cc1)O",  # Ibuprofen (Painkiller, NSAID)
        "COc1ccc(cc1)C(O)CN(C)C",  # Paracetamol (Painkiller, Fever reducer)
        "CCN1C(=O)c2c(ncn2C)n(c1=O)C",  # Caffeine (Stimulant)
        "CCOC(=O)c1c(cc(cc1O)O)C(=O)O",  # Aspirin (Painkiller, NSAID)
        "CN(C)C(=O)c1ccc(cc1)O",  # Acetaminophen (Analgesic)
        "COC(=O)c1c(nc(nc1O)N)N",  # Methotrexate (Chemotherapy, Autoimmune diseases)
        "CN(C)C(=O)c1c(cc(cc1O)O)O",  # Dopamine (Neurotransmitter, Parkinson's treatment)
        "O=C(O)c1ccccc1O",  # Salicylic Acid (Pain relief, Skincare)
    ]


@pytest.fixture
def fingerprints(simple_smiles, pharma_smiles):
    """Generate fingerprints for all molecules."""
    rdkit_simple = [smiles_to_morgan_fingerprint(s) for s in simple_smiles]
    rdkit_pharma = [smiles_to_morgan_fingerprint(s) for s in pharma_smiles]
    rdkit_all = rdkit_simple + rdkit_pharma

    np_simple = np.array([list(fp) for fp in rdkit_simple])
    np_pharma = np.array([list(fp) for fp in rdkit_pharma])
    np_all = np.array([list(fp) for fp in rdkit_all])

    pt_simple = torch.tensor(np_simple, dtype=torch.float32)
    pt_pharma = torch.tensor(np_pharma, dtype=torch.float32)
    pt_all = torch.tensor(np_all, dtype=torch.float32)

    return {
        'rdkit_simple': rdkit_simple,
        'rdkit_pharma': rdkit_pharma,
        'rdkit_all': rdkit_all,
        'np_simple': np_simple,
        'np_pharma': np_pharma,
        'np_all': np_all,
        'pt_simple': pt_simple,
        'pt_pharma': pt_pharma,
        'pt_all': pt_all
    }


@pytest.fixture
def ground_truths(fingerprints):
    """Generate ground truth similarity matrices."""
    return {
        'all': ground_truth(fingerprints['rdkit_all'], fingerprints['rdkit_all']),  # 18x18
        'simple_pharma': ground_truth(fingerprints['rdkit_simple'], fingerprints['rdkit_pharma']),  # 10x8
        'simple': ground_truth(fingerprints['rdkit_simple'], fingerprints['rdkit_simple']),  # 10x10
        'pharma': ground_truth(fingerprints['rdkit_pharma'], fingerprints['rdkit_pharma'])  # 8x8
    }


@pytest.mark.parametrize("method", ["rdkit", "torch", "numpy", "numpy-dot", "scipy", "numba", "torch-dot"])
@pytest.mark.parametrize("input_key,expected_key", [
    ("all_all", "all"),
    ("simple_pharma", "simple_pharma"),
    ("simple_simple", "simple"),
    ("pharma_pharma", "pharma")
])
def test_tanimoto_similarity(fingerprints, ground_truths, method, input_key, expected_key):
    """Test tanimoto similarity for all methods and input combinations."""
    # Map input keys to actual fingerprint pairs
    input_pairs = {
        "all_all": (fingerprints['rdkit_all'], fingerprints['rdkit_all']),
        "simple_pharma": (fingerprints['rdkit_simple'], fingerprints['rdkit_pharma']),
        "simple_simple": (fingerprints['rdkit_simple'], fingerprints['rdkit_simple']),
        "pharma_pharma": (fingerprints['rdkit_pharma'], fingerprints['rdkit_pharma'])
    }
    
    input_pair = input_pairs[input_key]
    expected = ground_truths[expected_key]
    
    output = tanimoto_similarity(*input_pair, method=method)
    
    # Check consistency
    np.testing.assert_allclose(
        output, expected, rtol=1e-6, atol=1e-6,
        err_msg=f"Method {method} failed for {input_key} (shape {expected.shape})"
    )


if __name__ == "__main__":
    pytest.main([__file__])