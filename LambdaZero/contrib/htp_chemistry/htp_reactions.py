from rdkit.Chem import AllChem

def rdkit_bicomponent_reactions_v0():
    # for reactions, the order matters, and every time only one substitution is done
    # fixme 1) need to include ester but not rings in suzuki, 2) need to include triflate, maybe recombine aldehyde/ketone
    amide_formation = AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[OH].[NX3;H2,H1;!$(NC=O):3][#6:4]>>[CX3:1](=[O:2])[*:3]([*:4])')
    esterification_alcohol = AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[OH:3].[OH][CX4:4]>>[CX3:1](=[O:2])[O:3][CX3:4]')
    esterification_halide = AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[OH:3].[Cl,Br,I][CX4:4]>>[CX3:1](=[O:2])[O:3][CX3:4]')
    reductive_amination_aldehyde = AllChem.ReactionFromSmarts('[#6:1][CX3H1:2](=O).[NX3;H2,H1;!$(NC=O):3][#6:4]>>[*:1][CX4H2:2][*:3]([*:4])')
    reductive_amination_ketone = AllChem.ReactionFromSmarts('[#6:1][CX3:2](=O)[#6:3].[NX3;H2,H1;!$(NC=O):4][#6:5]>>[*:1][CX4H1:2]([*:4][*:5])[*:3]') # maybe recombine
    buchwald_hartwig = AllChem.ReactionFromSmarts('[c:1][Cl,Br,I].[NX3;H2,H1;!$(NC=O):2][#6:3]>>[*:1][*:2]([*:3])')
    suzuki_coupling = AllChem.ReactionFromSmarts('[a:1][BX3](O)(O).[Cl,Br,I][c:2]>>[a:1][c:2]')

    return amide_formation, esterification_alcohol, esterification_halide, reductive_amination_aldehyde, reductive_amination_ketone, buchwald_hartwig, suzuki_coupling

def rdkit_monocomponent_reactions_v0():
    primary_alcohol_oxidation = AllChem.ReactionFromSmarts('[#6:1][CX4H2:2][OH]>>[*:1][CX3:2](=O)[OH]') # forms carboxylic acid
    secondary_alcohol_oxidation = AllChem.ReactionFromSmarts('[#6:1][CX4;H1,@H1,@@H1:2][OH]>>[*:1][CX3H0:2](=O)') # forms ketone, need to remove chirality
    nitrile_reduction = AllChem.ReactionFromSmarts('[CX2:1]#[NX1:2]>>[CX4H2:1][NH2:2]') # forms alkyl amines
    alkyl_halide_formation = AllChem.ReactionFromSmarts('[CX4:1][OH]>>[*:1][Br]') # forms alkyl halide
    borylation = AllChem.ReactionFromSmarts('[c:1][Cl,Br,I]>>[*:1]B(O)(O)') # forms aryl boronic acid
    # aryl halide is difficult to predict position, alkyl alcohol formation SN2/E2

    return primary_alcohol_oxidation, secondary_alcohol_oxidation, nitrile_reduction, alkyl_halide_formation, borylation
