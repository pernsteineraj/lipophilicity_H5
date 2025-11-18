# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:51:51 2025

@author: ajp07
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

def generate_maccs_keys(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return [MACCSkeys.GenMACCSKeys(m) for m in mols if m is not None]

def generate_morgan_fingerprints(smiles_list, radius=2, nbits=1024):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) 
            for m in smiles_list if m is not None]