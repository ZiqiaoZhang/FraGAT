import torch as t
from FraGAT.ChemUtils import *
import numpy as np
import random

class BasicFeaturizer(object):
    def __init__(self):
        super(BasicFeaturizer, self).__init__()

    def featurize(self, item):
        raise NotImplementedError(
            "Molecule Featurizer not implemented.")


class FPFeaturizer(BasicFeaturizer):
    def __init__(self, opt):
        super(FPFeaturizer, self).__init__()
        self.opt = opt

    def featurize(self, item):
        SMILES = item['SMILES']
        Value = item['Value']
        FP = GetMolFingerprints(SMILES, self.opt.args['nBits'])
        FP = t.Tensor(FP)
        Label = t.Tensor([int(Value)]).long()
        return FP, Label

class GraphFeaturizer(BasicFeaturizer):
    def __init__(self):
        super(GraphFeaturizer, self).__init__()

    def featurize(self, item):
        SMILES =item['SMILES']
        Value = item['Value']
        mol = Chem.MolFromSmiles(SMILES)
        AdjMat = GetAdjMat(mol)
        FeatureMat = GetMolFeatureMat(mol)
        AdjMat = t.Tensor(AdjMat)
        FeatureMat = t.Tensor(FeatureMat)
        Label = t.Tensor([int(Value)]).long()

        ids = self.zero_degree_check(AdjMat)
        AdjMat = t.index_select(AdjMat, dim=0, index=ids)
        AdjMat = t.index_select(AdjMat, dim=1, index=ids)
        FeatureMat = t.index_select(FeatureMat, dim=0, index=ids)

        return (AdjMat, FeatureMat), Label

    def zero_degree_check(self, AdjMat):
        degree = t.sum(AdjMat, dim=1)   # t.sum(), calculate sum on dim x, which means that the output tensor on dim x will disappear or squeeze to 1
        ids = [i for i,x in enumerate(degree) if x != 0]
        ids = t.Tensor(ids).long()
        return ids

class AttentiveFPFeaturizer(BasicFeaturizer):
    def __init__(self, atom_feature_size, bond_feature_size, max_degree, max_frag, mode):
        super(AttentiveFPFeaturizer, self).__init__()
        self.max_atom_num = 0
        self.max_bond_num = 0
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.max_degree = max_degree
        self.mode = mode
        self.max_frag = max_frag

    def featurize(self, dataset, index, mol, value, Frag):
        [entire_atom_features, entire_bond_features, entire_atom_neighbor_list, entire_bond_neighbor_list, entire_atom_mask] = dataset

        mol_atom_features = entire_atom_features[index]
        mol_bond_features = entire_bond_features[index]
        mol_atom_neighbor_list = entire_atom_neighbor_list[index]
        mol_bond_neighbor_list = entire_bond_neighbor_list[index]
        mol_atom_mask = entire_atom_mask[index]

        mol_atom_features = t.Tensor(mol_atom_features)
        mol_bond_features = t.Tensor(mol_bond_features)
        mol_atom_neighbor_list = t.Tensor(mol_atom_neighbor_list).long()
        mol_bond_neighbor_list = t.Tensor(mol_bond_neighbor_list).long()
        mol_atom_mask = t.Tensor(mol_atom_mask)

        label = []
        for v in value:
            label.append(int(v))

        label = t.Tensor(label).long()
        label.unsqueeze_(-1)

        # sizes:
        # mol_atom_features: [max_atom_length, atom_feature_size]
        # mol_bond_features: [max_bond_length, bond_feature_size]
        # mol_atom_neighbor_list: [max_atom_length, max_degree]
        # mol_bond_neighbor_list: [max_atom_length, max_degree]
        # mol_atom_mask: [max_atom_length]

        if Frag:
            if self.mode == "TRAIN":
                mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx = self.CutSingleBond(
                    mol, mol_atom_neighbor_list, mol_bond_neighbor_list)
                if bond_idx:
                    mask1, mask2 = self.GetComponentMasks(start_atom, end_atom, mol_atom_neighbor_list_changed)
                    mask1 = t.Tensor(mask1)
                    mask2 = t.Tensor(mask2)
                    mol_atom_mask1 = mask1 * mol_atom_mask
                    mol_atom_mask2 = mask2 * mol_atom_mask
                    bond_idx = t.Tensor([bond_idx]).long()
                #print(bond_idx.size())
                #print(bond_idx)
                    return [mol_atom_features,
                            mol_bond_features,
                            mol_atom_neighbor_list_changed,
                            mol_bond_neighbor_list_changed,
                            mol_atom_mask1,
                            mol_atom_mask2,
                            bond_idx,
                            mol_atom_neighbor_list,
                            mol_bond_neighbor_list,
                            mol_atom_mask], label

                else:
                    # No single bonds can be cut.
                    return [mol_atom_features,
                            mol_bond_features,
                            mol_atom_neighbor_list_changed,
                            mol_bond_neighbor_list_changed,
                            mol_atom_mask,
                            mol_atom_mask,
                            t.Tensor([-1]).long(),
                            mol_atom_neighbor_list,
                            mol_bond_neighbor_list,
                            mol_atom_mask], label
            elif self.mode == 'EVAL':
                extended_atom_features = t.Tensor([])
                extended_bond_features = t.Tensor([])
                extended_atom_neighbor_list_changed = t.Tensor([]).long()
                extended_bond_neighbor_list_changed = t.Tensor([]).long()
                extended_atom_mask1 = t.Tensor([])
                extended_atom_mask2 = t.Tensor([])
                extended_bond_idx = t.Tensor([]).long()
                extended_atom_neighbor_list = t.Tensor([]).long()
                extended_bond_neighbor_list = t.Tensor([]).long()
                extended_atom_mask = t.Tensor([])
                extended_label = t.Tensor([]).long()

                SingleBondList = GetSingleBonds(mol)
                assert len(SingleBondList) > 0
                for bond in SingleBondList:
                    mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx \
                        = self.CutOneBond(bond, mol_atom_neighbor_list, mol_bond_neighbor_list)
                    if True:
                        mask1, mask2 = self.GetComponentMasks(start_atom, end_atom, mol_atom_neighbor_list_changed)
                        mask1 = t.Tensor(mask1)
                        mask2 = t.Tensor(mask2)
                        mol_atom_mask1 = mask1 * mol_atom_mask
                        mol_atom_mask2 = mask2 * mol_atom_mask
                        bond_idx = t.Tensor([bond_idx]).long()
                        # print(bond_idx.size())
                        # print(bond_idx)
                        extended_atom_features = self.CatTensor(extended_atom_features, mol_atom_features)
                        extended_bond_features = self.CatTensor(extended_bond_features, mol_bond_features)
                        extended_atom_neighbor_list_changed = self.CatTensor(extended_atom_neighbor_list_changed, mol_atom_neighbor_list_changed)
                        extended_bond_neighbor_list_changed = self.CatTensor(extended_bond_neighbor_list_changed, mol_bond_neighbor_list_changed)
                        extended_atom_mask1 = self.CatTensor(extended_atom_mask1, mol_atom_mask1)
                        extended_atom_mask2 = self.CatTensor(extended_atom_mask2, mol_atom_mask2)
                        extended_bond_idx = self.CatTensor(extended_bond_idx, bond_idx)
                        extended_atom_neighbor_list = self.CatTensor(extended_atom_neighbor_list, mol_atom_neighbor_list)
                        extended_bond_neighbor_list = self.CatTensor(extended_bond_neighbor_list, mol_bond_neighbor_list)
                        extended_atom_mask = self.CatTensor(extended_atom_mask, mol_atom_mask)
                        extended_label = self.CatTensor(extended_label, label)

                return [extended_atom_features,
                        extended_bond_features,
                        extended_atom_neighbor_list_changed,
                        extended_bond_neighbor_list_changed,
                        extended_atom_mask1,
                        extended_atom_mask2,
                        extended_bond_idx,
                        extended_atom_neighbor_list,
                        extended_bond_neighbor_list,
                        extended_atom_mask], extended_label

            else:
                print("Wrong mode.")
                raise RuntimeError

        return [mol_atom_features, mol_bond_features, mol_atom_neighbor_list, mol_bond_neighbor_list, mol_atom_mask], label

    def featurizenew(self, dataset, index, mol, value, Frag, opt):
        [entire_atom_features, entire_bond_features, entire_atom_neighbor_list, entire_bond_neighbor_list, entire_atom_mask] = dataset

        mol_atom_features = entire_atom_features[index]
        mol_bond_features = entire_bond_features[index]
        mol_atom_neighbor_list = entire_atom_neighbor_list[index]
        mol_bond_neighbor_list = entire_bond_neighbor_list[index]
        mol_atom_mask = entire_atom_mask[index]

        mol_atom_features = t.Tensor(mol_atom_features)
        mol_bond_features = t.Tensor(mol_bond_features)
        mol_atom_neighbor_list = t.Tensor(mol_atom_neighbor_list).long()
        mol_bond_neighbor_list = t.Tensor(mol_bond_neighbor_list).long()
        mol_atom_mask = t.Tensor(mol_atom_mask)

        label = []
        if opt.args['ClassNum'] == 1:
            if opt.args['TaskNum'] == 1:
                label.append(float(value))
            else:
                for v in value:
                    label.append(float(v))
            label = t.Tensor(label)
        else:
            for v in value:
                label.append(int(v))
            label = t.Tensor(label).long()

        label.unsqueeze_(-1)

        # sizes:
        # mol_atom_features: [max_atom_length, atom_feature_size]
        # mol_bond_features: [max_bond_length, bond_feature_size]
        # mol_atom_neighbor_list: [max_atom_length, max_degree]
        # mol_bond_neighbor_list: [max_atom_length, max_degree]
        # mol_atom_mask: [max_atom_length]

        if Frag:
            if self.mode == "TRAIN":
                # create the information of one molecule.
                mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx = self.CutSingleBond(
                    mol, mol_atom_neighbor_list, mol_bond_neighbor_list)
                # No matter whether a bond has been cut, the structure of the return are the same.
                # However, if no bond is cut, the two neighbor_list_changed are the same as the original neighbor lists.
                # and the start_atom, end_atom, bond_idx are None.

                if bond_idx:
                    mask1, mask2 = self.GetComponentMasks(start_atom, end_atom, mol_atom_neighbor_list_changed)
                    mask1 = t.Tensor(mask1)
                    mask2 = t.Tensor(mask2)
                    mol_frag_mask1 = mask1 * mol_atom_mask
                    mol_frag_mask2 = mask2 * mol_atom_mask
                    # In the atom_neighbor_list, all atoms are set to be connected with the pad node.
                    # so that the generated mask1 and mask2 are not correct because the pad nodes are set to 1.
                    # That's why we should use mask1 * mol_atom_mask to set the pad nodes to 0.

                    bond_idx = t.Tensor([bond_idx]).long()

                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=start_atom, end_atom=end_atom, bondidx=bond_idx, frag_masks=[mol_frag_mask1, mol_frag_mask2])
                    # Return in such format: Origin Graph, Frags, Junction Tree
                    return [mol_atom_features,
                            mol_bond_features,
                            mol_atom_neighbor_list,
                            mol_bond_neighbor_list,
                            mol_atom_mask,
                            mol_atom_neighbor_list_changed,
                            mol_bond_neighbor_list_changed,
                            mol_frag_mask1,
                            mol_frag_mask2,
                            bond_idx,
                            JT_bond_features,
                            JT_atom_neighbor_list,
                            JT_bond_neighbor_list,
                            JT_mask], label

                else:
                    # No single bonds can be cut.
                    # Return in such format: Origin Graph, Frags, Junction Tree
                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=None, end_atom=None, bondidx=[], frag_masks=[])
                    return [mol_atom_features,
                            mol_bond_features,
                            mol_atom_neighbor_list,
                            mol_bond_neighbor_list,
                            mol_atom_mask,
                            mol_atom_neighbor_list_changed,
                            mol_bond_neighbor_list_changed,
                            mol_atom_mask,
                            t.zeros(self.max_atom_num),
                            t.Tensor([-1]).long(),
                            JT_bond_features,
                            JT_atom_neighbor_list,
                            JT_bond_neighbor_list,
                            JT_mask], label

            elif self.mode == 'EVAL':
                # create a 'batch' of molecules
                extended_atom_features = t.Tensor([])
                extended_bond_features = t.Tensor([])
                extended_atom_neighbor_list = t.Tensor([]).long()
                extended_bond_neighbor_list = t.Tensor([]).long()
                extended_atom_mask = t.Tensor([])

                extended_atom_neighbor_list_changed = t.Tensor([]).long()
                extended_bond_neighbor_list_changed = t.Tensor([]).long()
                extended_frag_mask1 = t.Tensor([])
                extended_frag_mask2 = t.Tensor([])
                extended_bond_idx = t.Tensor([]).long()

                extended_JT_bond_features = t.Tensor([])
                extended_JT_atom_neighbor_list = t.Tensor([]).long()
                extended_JT_bond_neighbor_list = t.Tensor([]).long()
                extended_JT_mask = t.Tensor([])

                if opt.args['ClassNum'] == 1:
                    extended_label = t.Tensor([])
                else:
                    extended_label = t.Tensor([]).long()

                SingleBondList = GetSingleBonds(mol)
                #assert len(SingleBondList) > 0
                if len(SingleBondList) == 0:
                    # No bond is cut. Only one molecule information is used.
                    # Original Graph, no change
                    extended_atom_features = self.CatTensor(extended_atom_features, mol_atom_features)
                    extended_bond_features = self.CatTensor(extended_bond_features, mol_bond_features)
                    extended_atom_neighbor_list = self.CatTensor(extended_atom_neighbor_list, mol_atom_neighbor_list)
                    extended_bond_neighbor_list = self.CatTensor(extended_bond_neighbor_list, mol_bond_neighbor_list)
                    extended_atom_mask = self.CatTensor(extended_atom_mask, mol_atom_mask)

                    # Frags, no change.
                    extended_atom_neighbor_list_changed = self.CatTensor(extended_atom_neighbor_list_changed, mol_atom_neighbor_list)
                    extended_bond_neighbor_list_changed = self.CatTensor(extended_bond_neighbor_list_changed, mol_bond_neighbor_list)
                    extended_frag_mask1 = self.CatTensor(extended_frag_mask1, mol_atom_mask)
                    extended_frag_mask2 = self.CatTensor(extended_frag_mask2, t.zeros(self.max_atom_num))
                    extended_bond_idx = self.CatTensor(extended_bond_idx, t.Tensor([-1]).long())

                    # Junction Tree
                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=None,
                        end_atom=None, bondidx=[], frag_masks=[])

                    extended_JT_bond_features = self.CatTensor(extended_JT_bond_features, JT_bond_features)
                    extended_JT_atom_neighbor_list = self.CatTensor(extended_JT_atom_neighbor_list, JT_atom_neighbor_list)
                    extended_JT_bond_neighbor_list = self.CatTensor(extended_JT_bond_neighbor_list, JT_bond_neighbor_list)
                    extended_JT_mask = self.CatTensor(extended_JT_mask, JT_mask)

                    extended_label = self.CatTensor(extended_label, label)


                else:
                    for bond in SingleBondList:
                        # Cut one bond
                        mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx \
                            = self.CutOneBond(bond, mol_atom_neighbor_list, mol_bond_neighbor_list)
                        #if True:
                        mask1, mask2 = self.GetComponentMasks(start_atom, end_atom, mol_atom_neighbor_list_changed)
                        mask1 = t.Tensor(mask1)
                        mask2 = t.Tensor(mask2)
                        mol_frag_mask1 = mask1 * mol_atom_mask
                        mol_frag_mask2 = mask2 * mol_atom_mask
                        bond_idx = t.Tensor([bond_idx]).long()
                            # print(bond_idx.size())
                            # print(bond_idx)
                        JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                            mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=start_atom,
                            end_atom=end_atom, bondidx=bond_idx, frag_masks=[mol_frag_mask1, mol_frag_mask2])

                        extended_atom_features = self.CatTensor(extended_atom_features, mol_atom_features)
                        extended_bond_features = self.CatTensor(extended_bond_features, mol_bond_features)
                        extended_atom_neighbor_list = self.CatTensor(extended_atom_neighbor_list,
                                                                     mol_atom_neighbor_list)
                        extended_bond_neighbor_list = self.CatTensor(extended_bond_neighbor_list,
                                                                     mol_bond_neighbor_list)
                        extended_atom_mask = self.CatTensor(extended_atom_mask, mol_atom_mask)

                        extended_atom_neighbor_list_changed = self.CatTensor(extended_atom_neighbor_list_changed, mol_atom_neighbor_list_changed)
                        extended_bond_neighbor_list_changed = self.CatTensor(extended_bond_neighbor_list_changed, mol_bond_neighbor_list_changed)
                        extended_frag_mask1 = self.CatTensor(extended_frag_mask1, mol_frag_mask1)
                        extended_frag_mask2 = self.CatTensor(extended_frag_mask2, mol_frag_mask2)
                        extended_bond_idx = self.CatTensor(extended_bond_idx, bond_idx)

                        extended_JT_bond_features = self.CatTensor(extended_JT_bond_features, JT_bond_features)
                        extended_JT_atom_neighbor_list = self.CatTensor(extended_JT_atom_neighbor_list,
                                                                        JT_atom_neighbor_list)
                        extended_JT_bond_neighbor_list = self.CatTensor(extended_JT_bond_neighbor_list,
                                                                        JT_bond_neighbor_list)
                        extended_JT_mask = self.CatTensor(extended_JT_mask, JT_mask)

                        extended_label = self.CatTensor(extended_label, label)

                return [extended_atom_features,
                        extended_bond_features,
                        extended_atom_neighbor_list,
                        extended_bond_neighbor_list,
                        extended_atom_mask,
                        extended_atom_neighbor_list_changed,
                        extended_bond_neighbor_list_changed,
                        extended_frag_mask1,
                        extended_frag_mask2,
                        extended_bond_idx,
                        extended_JT_bond_features,
                        extended_JT_atom_neighbor_list,
                        extended_JT_bond_neighbor_list,
                        extended_JT_mask], extended_label

            else:
                print("Wrong mode.")
                raise RuntimeError

        return [mol_atom_features, mol_bond_features, mol_atom_neighbor_list, mol_bond_neighbor_list, mol_atom_mask], label

    def CreateJunctionTree(self, mol_bond_features, atom_neighbor_list, bond_neighbor_list, start_atom, end_atom, bondidx, frag_masks):
        # For simple, in this version, we only consider that 0 or 1 bond is cut.
        # The case that multiple bonds are cut will be considered in the furture version.
        pad_bond_feature = t.zeros(1, self.bond_feature_size)        # [1, bond_feature_size]
        cut_bonds_num = len(bondidx)
        if cut_bonds_num == 0:
            JT_bond_feature = t.cat([pad_bond_feature, pad_bond_feature])      # [2, bond_feature_size]. max_bond_num = 2

            JT_atom_neighbor_list = np.zeros([3, self.max_degree])                # max_frag_num = 2, max_atom_num = 3, for 1 pad node.
            JT_atom_neighbor_list.fill(2)
            JT_bond_neighbor_list = np.zeros([3, self.max_degree])
            JT_bond_neighbor_list.fill(1)

            JT_atom_neighbor_list = t.Tensor(JT_atom_neighbor_list).long()
            JT_bond_neighbor_list = t.Tensor(JT_bond_neighbor_list).long()

            JT_mask = t.Tensor([1.0,0.0,0.0])

        elif cut_bonds_num == 1:
            JT_bond_feature = t.cat([mol_bond_features[bondidx], pad_bond_feature])        # [2, bond_feature_size]
            JT_atom_neighbor_list = np.zeros([3, self.max_degree])  # max_frag_num = 2, max_atom_num = 3, for 1 pad node.
            JT_atom_neighbor_list.fill(2)
            JT_bond_neighbor_list = np.zeros([3, self.max_degree])
            JT_bond_neighbor_list.fill(1)

            JT_atom_neighbor_list[0,0] = 1
            JT_atom_neighbor_list[1,0] = 0
            JT_bond_neighbor_list[0,0] = 0
            JT_bond_neighbor_list[1,0] = 0

            JT_atom_neighbor_list = t.Tensor(JT_atom_neighbor_list).long()
            JT_bond_neighbor_list = t.Tensor(JT_bond_neighbor_list).long()

            JT_mask = t.Tensor([1.0,1.0,0.0])
        # sizes:
        # mol_atom_features: [max_atom_length, atom_feature_size]
        # mol_bond_features: [max_bond_length, bond_feature_size]
        # mol_atom_neighbor_list: [max_atom_length, max_degree]
        # mol_bond_neighbor_list: [max_atom_length, max_degree]
        # mol_atom_mask: [max_atom_length]
        return JT_bond_feature, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask


    def prefeaturize(self, dataset):
        entire_atom_features = []
        entire_bond_features = []
        entire_atom_neighbor_list = []
        entire_bond_neighbor_list = []
        entire_atom_mask = []

        for item in dataset:
            SMILES = item['SMILES']
            mol = Chem.MolFromSmiles(SMILES)

            mol_atom_feature = np.zeros([self.max_atom_num, self.atom_feature_size])
            mol_bond_feature = np.zeros([self.max_bond_num, self.bond_feature_size])

            mol_atom_neighbor_list = np.zeros([self.max_atom_num, self.max_degree])
            mol_bond_neighbor_list = np.zeros([self.max_atom_num, self.max_degree])
            mol_atom_neighbor_list.fill(self.pad_atom_idx)
            mol_bond_neighbor_list.fill(self.pad_bond_idx)

            mol_atom_mask = np.zeros([self.max_atom_num])

            #  generate five information of a molecule.

            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                mol_atom_mask[idx] = 1.0
                atom_feature = GetAtomFeatures(atom)
                mol_atom_feature[idx] = atom_feature

                neighbors = atom.GetNeighbors()
                pointer = 0
                for neighbor in neighbors:
                    nei_idx = neighbor.GetIdx()
                    mol_atom_neighbor_list[idx][pointer] = nei_idx
                    pointer += 1

            bond_pointer = np.zeros([self.max_atom_num])
            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                bond_feature = GetBondFeatures(bond)
                mol_bond_feature[idx] = bond_feature

                start_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()

                start_atom_pointer = int(bond_pointer[start_atom])
                end_atom_pointer = int(bond_pointer[end_atom])

                mol_bond_neighbor_list[start_atom][start_atom_pointer] = idx
                mol_bond_neighbor_list[end_atom][end_atom_pointer] = idx

                bond_pointer[start_atom] += 1
                bond_pointer[end_atom] += 1

            entire_atom_features.append(mol_atom_feature)
            entire_bond_features.append(mol_bond_feature)
            entire_atom_neighbor_list.append(mol_atom_neighbor_list)
            entire_bond_neighbor_list.append(mol_bond_neighbor_list)
            entire_atom_mask.append(mol_atom_mask)

        return [entire_atom_features, entire_bond_features, entire_atom_neighbor_list, entire_bond_neighbor_list, entire_atom_mask]

    def GetPad(self, dataset):
        # dataset format: [{"SMILES": smiles, "Value": value}]

        for item in dataset:
            smiles = item["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            total_atom_num = len(mol.GetAtoms())
            total_bond_num = len(mol.GetBonds())
            self.max_atom_num = max(self.max_atom_num, total_atom_num)
            self.max_bond_num = max(self.max_bond_num, total_bond_num)

        self.pad_atom_idx = self.max_atom_num
        self.pad_bond_idx = self.max_bond_num

        self.max_atom_num += 1
        self.max_bond_num += 1

    def CatTensor(self, stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = t.cat([stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

    def CutOneBond(self, bond, mol_atom_neighbor_list, mol_bond_neighbor_list):   # for eval
        _mol_atom_neighbor_list = mol_atom_neighbor_list.clone()
        _mol_bond_neighbor_list = mol_bond_neighbor_list.clone()
        # insulate.
        [bond_idx, start_atom_idx, end_atom_idx] = bond
        assert end_atom_idx in _mol_atom_neighbor_list[start_atom_idx]
        assert start_atom_idx in _mol_atom_neighbor_list[end_atom_idx]
        # print(start_atom_idx)
        # print(end_atom_idx)
        # print(bond_idx)

        loc = _mol_atom_neighbor_list[start_atom_idx].tolist().index(end_atom_idx)
        _mol_atom_neighbor_list[start_atom_idx][loc] = self.pad_atom_idx
        loc = _mol_atom_neighbor_list[end_atom_idx].tolist().index(start_atom_idx)
        _mol_atom_neighbor_list[end_atom_idx][loc] = self.pad_atom_idx

        loc = _mol_bond_neighbor_list[start_atom_idx].tolist().index(bond_idx)
        _mol_bond_neighbor_list[start_atom_idx][loc] = self.pad_bond_idx
        loc = _mol_bond_neighbor_list[end_atom_idx].tolist().index(bond_idx)
        _mol_bond_neighbor_list[end_atom_idx][loc] = self.pad_bond_idx

        return _mol_atom_neighbor_list, _mol_bond_neighbor_list, start_atom_idx, end_atom_idx, bond_idx

    def CutSingleBond(self, mol, mol_atom_neighbor_list, mol_bond_neighbor_list):   # for train
        # This function will calculate the SingleBondList and tries to cut a random one.
        # if len(SingleBondList) > 0, one single bond will be cut. The two neighbor lists will be modified.
        # the return is [mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx]
        # and if len(SingleBondList) == 0, no single bond will be cut. The two neighbor lists will not be modified.
        # the return is [mol_atom_neighbor_list, mol_bond_neighbor_list, None, None, None]
        # This function is compatible with the molecules that cannot be cut.

        # mol_atom_neighbor_list and mol_bond_neighbor_list are original neighbor lists that transmit to this function.
        # However, using neighbor_list[x,x] = xxx will exactly change the value of the original neighbor lists.
        # so ,the Tensors should be cloned first, to make sure that the Tensors outside of this function will not be changed.
        _mol_atom_neighbor_list = mol_atom_neighbor_list.clone()
        _mol_bond_neighbor_list = mol_bond_neighbor_list.clone()
        # insulate.
        SingleBondList = GetSingleBonds(mol)
        if len(SingleBondList) > 0:

            # Choose one bond to cut.
            random.shuffle(SingleBondList)
            [bond_idx, start_atom_idx, end_atom_idx] = SingleBondList[0]
            assert end_atom_idx in _mol_atom_neighbor_list[start_atom_idx]
            assert start_atom_idx in _mol_atom_neighbor_list[end_atom_idx]

            # modify the two neighbor lists based on the chosen bond.
            loc = _mol_atom_neighbor_list[start_atom_idx].tolist().index(end_atom_idx)
            _mol_atom_neighbor_list[start_atom_idx][loc] = self.pad_atom_idx
            loc = _mol_atom_neighbor_list[end_atom_idx].tolist().index(start_atom_idx)
            _mol_atom_neighbor_list[end_atom_idx][loc] = self.pad_atom_idx

            loc = _mol_bond_neighbor_list[start_atom_idx].tolist().index(bond_idx)
            _mol_bond_neighbor_list[start_atom_idx][loc] = self.pad_bond_idx
            loc = _mol_bond_neighbor_list[end_atom_idx].tolist().index(bond_idx)
            _mol_bond_neighbor_list[end_atom_idx][loc] = self.pad_bond_idx

            return _mol_atom_neighbor_list, _mol_bond_neighbor_list, start_atom_idx, end_atom_idx, bond_idx
        else:
            # no bond can be cut. _nei_list is same as the original one
            return _mol_atom_neighbor_list, _mol_bond_neighbor_list, None, None, None

    def GetComponentMasks(self, root_node1, root_node2, mol_atom_neighbor_list):
        mask1 = self.ComponentSearch(mol_atom_neighbor_list, self.max_atom_num, root_node1)
        mask2 = self.ComponentSearch(mol_atom_neighbor_list, self.max_atom_num, root_node2)
        assert len(mask1) == len(mask2)
        return mask1, mask2

    def ComponentSearch(self, mol_atom_neighbor_list, max_atom_num, root_node):
        candidate_set = []
        mask = np.zeros([max_atom_num])
        mask[root_node] = 1
        candidate_set.append(root_node)

        while len(candidate_set) > 0:
            node = candidate_set[0]
            candidate_set.pop(0)

            neighbors = mol_atom_neighbor_list[node]
            for nei in neighbors:
                if mask[nei] == 0:
                    candidate_set.append(nei)
                    mask[nei] = 1

        assert len(candidate_set) == 0
        return mask














