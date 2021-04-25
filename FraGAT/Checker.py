import rdkit.Chem as Chem

class BasicChecker(object):
    def __init__(self):
        super(BasicChecker, self).__init__()

    def check(self, dataset):
        raise NotImplementedError(
            "Dataset Checker not implemented.")

class ScaffoldSplitterChecker(BasicChecker):
    def __init__(self):
        super(ScaffoldSplitterChecker, self).__init__()

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                checked_dataset.append(item)
            else:
                discarded_dataset.append(item)
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")

        return checked_dataset

class AttentiveFPChecker(BasicChecker):
    def __init__(self, max_atom_num, max_degree):
        super(AttentiveFPChecker, self).__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            #check
            if mol:
                #self.check_single_bonds(mol)
                self.check_degree(mol)
                self.check_max_atom_num(mol)

                if self.mol_error_flag == 0:
                    checked_dataset.append(item)
                else:
                    discarded_dataset.append(item)
                    self.mol_error_flag = 0
            else:
                discarded_dataset.append(item)
                self.mol_error_flag = 0
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")

        return checked_dataset

    def check_degree(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetDegree() > self.max_degree:
                self.mol_error_flag = 1
                break

    def check_max_atom_num(self, mol):
        if len(mol.GetAtoms()) > self.max_atom_num:
            self.mol_error_flag = 1
    def check_single_bonds(self, mol):
        self.mol_error_flag = 1
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if not bond.IsInRing():
                    self.mol_error_flag = 0
                    break
