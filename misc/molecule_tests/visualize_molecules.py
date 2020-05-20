from rdkit import Chem
from rdkit.Chem import Draw

class cfg:

    smis = ["CN(C)C1CCN(C2CCCN2CN(CNCN)c2cnn(Cl)c2F)CC1", "CC=Cc1cc(CN)nn1C(N)N1CCC(N(C)C)C1N1CCCC1N",
            "CN1CCN(CN(CCC#N)C2CCC(NCc3ccccc3)N2)CC1", "CNCn1nc(C2CCC(N3CCN(CCF)CC3)N2CN)cc1Cl",
            "C=Cc1cn(CNC2CCN(CNC)C2N2CCN(C)CC2)nc1Cl", "CCCN1CCCC1N1CCN(C2CCC(N3CCC(C[NH3+])C3)N2)CC1"]



mols = [Chem.MolFromSmiles(smi) for smi in cfg.smis]

img=Draw.MolsToGridImage(mols,molsPerRow=3,subImgSize=(500,500))
img.save("/home/maksym/Desktop/mols.png")
#Draw.MolToFile()