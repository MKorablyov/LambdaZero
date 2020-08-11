from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.rdMolAlign import AlignMol


def MyConstrainedEmbed(mol, core, coreConfId=-1, randomseed=2342,
                       getForceField=UFFGetMoleculeForceField, **kwargs):
    """
    This is a copy of ConstrainedEmbed where I make the constraints more solid.
    """
    energy_tol = 1e-10
    force_tol = 1e-10
    force_constant = 100000.

    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    ci = EmbedMultipleConfs(mol, numConfs=50, coordMap=coordMap, randomSeed=randomseed, **kwargs)

    algMap = [(j, i) for i, j in enumerate(match)]

    # clean up the conformation
    for conf_id in [c.GetId() for c in mol.GetConformers()]:
        ff = getForceField(mol, confId=conf_id)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, force_constant)
                #ff.MMFFAddDistanceConstraint(idxI, idxJ, d, d, force_constant)

        ff.Initialize()
        counter = 0
        more = 1
        while more:
            more = ff.Minimize(energyTol=energy_tol, forceTol=force_tol)
            counter += 1
            if counter > 10:
                print("More than 10 iterations")
                break
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol, core, prbCid=conf_id, refCid=coreConfId, atomMap=algMap)
        print(rms)



    mol.SetProp('EmbedRMS', str(rms))
    return mol