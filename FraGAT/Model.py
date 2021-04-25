import torch as t
import torch.nn as nn
import numpy as np

from FraGAT.ClassifierModel import DNN
#from FraGAT.NaiveGCN import NaiveGCN
from FraGAT.AttentiveFP import *

'''
class MolPredGraph(nn.Module):
    def __init__(self, feature_size, GCN_layers, DNN_layers, GCNLayertype):
        super(MolPredGraph, self).__init__()
        self.GCN = NaiveGCN(feature_size = feature_size, layers = GCN_layers, GCNLayertype = GCNLayertype)
        self.Classifier = DNN(input_size = 128, layer_sizes = DNN_layers, output_size = 2)
        self.Linear = nn.Linear(GCN_layers[-1], 128)

    def readout(self, FeatureMat):
        MolFeat = t.mean(FeatureMat, dim=-2)
        MolFeat = self.Linear(MolFeat)
        return MolFeat
        #return t.mean(FeatureMat, dim=-2)

    def forward(self, Input):
        [AdjMat, FeatureMat] = Input
        AdjMat = AdjMat.cuda()
        FeatureMat = FeatureMat.cuda()
        X = self.GCN(AdjMat, FeatureMat)           # [batch, atom_num, feature_num]
        X = self.readout(X)                        # [batch, feature_num]
        X = self.Classifier(X)
        return X


class MolPredFP(nn.Module):
    def __init__(self, feature_size, DNN_layers):
        super(MolPredFP, self).__init__()
        self.Classifier = DNN(input_size = feature_size, layer_sizes = DNN_layers, output_size = 2)
    def forward(self, Input):
        X = Input.cuda()
        X = self.Classifier(X)
        return X
'''
class MolPredFragFPv8(nn.Module):
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 drop_rate,
                 opt
                 ):
        super(MolPredFragFPv8, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size = atom_feature_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )   # For Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=drop_rate
        )  # MolEmbedding module can be used repeatedly
        self.Classifier = DNN(
                input_size=4*FP_size,
                output_size=output_size,
                layer_sizes=DNN_layers,
                opt = opt
        )
        self.AtomEmbeddingHigher = AttentiveFP_atom(
            atom_feature_size = FP_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )  # For Junction Tree
        #self.InformationFuser =

    def forward(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = self.Input_cuda(Input)

        # layer origin
        atom_FP_origin = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_origin, bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # layer Frag:
        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are used to input the DNN module.
        # activated_mol_FP1 and activated_mol_FP2 are used to calculate the mol_FP
        # size: [batch_size, FP_size]
        ##################################################################################
        # Junction Tree Construction
        # construct a higher level graph: Junction Tree

        # Construct atom features of JT:
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = t.zeros(batch_size, FP_size).cuda()
        JT_atom_features = t.stack([activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim = 1)

        # Junction Tree Construction complete.
        ##################################################################################
        #layer Junction Tree: calculate information of the junction tree of Frags
        atom_FP_super = self.AtomEmbeddingHigher(JT_atom_features,
                                           JT_bond_features,
                                           JT_atom_neighbor_list,
                                           JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)

        entire_FP = t.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)
        prediction = self.Classifier(entire_FP)
        return prediction

    def Input_cuda(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input
        if not self.training:
            #print(atom_features.size())
            atom_features = atom_features.squeeze(dim=0).cuda()
            bond_features = bond_features.squeeze(dim=0).cuda()
            atom_neighbor_list_origin = atom_neighbor_list_origin.squeeze(dim=0).cuda()
            bond_neighbor_list_origin = bond_neighbor_list_origin.squeeze(dim=0).cuda()
            atom_mask_origin = atom_mask_origin.squeeze(dim=0).cuda()

            atom_neighbor_list_changed = atom_neighbor_list_changed.squeeze(dim=0).cuda()
            bond_neighbor_list_changed = bond_neighbor_list_changed.squeeze(dim=0).cuda()
            frag_mask1 = frag_mask1.squeeze(dim=0).cuda()
            frag_mask2 = frag_mask2.squeeze(dim=0).cuda()
            bond_index = bond_index.squeeze(dim=0).cuda()

            JT_bond_features = JT_bond_features.squeeze(dim=0).cuda()
            JT_atom_neighbor_list = JT_atom_neighbor_list.squeeze(dim=0).cuda()
            JT_bond_neighbor_list = JT_bond_neighbor_list.squeeze(dim=0).cuda()
            JT_mask = JT_mask.squeeze(dim=0).cuda()

        else:
            atom_features = atom_features.cuda()
            bond_features = bond_features.cuda()
            atom_neighbor_list_origin = atom_neighbor_list_origin.cuda()
            bond_neighbor_list_origin = bond_neighbor_list_origin.cuda()
            atom_mask_origin = atom_mask_origin.cuda()

            atom_neighbor_list_changed = atom_neighbor_list_changed.cuda()
            bond_neighbor_list_changed = bond_neighbor_list_changed.cuda()
            frag_mask1 = frag_mask1.cuda()
            frag_mask2 = frag_mask2.cuda()
            bond_index = bond_index.cuda()

            JT_bond_features = JT_bond_features.cuda()
            JT_atom_neighbor_list = JT_atom_neighbor_list.cuda()
            JT_bond_neighbor_list = JT_bond_neighbor_list.cuda()
            JT_mask = JT_mask.cuda()
        return [atom_features,
                bond_features,
                atom_neighbor_list_origin,
                bond_neighbor_list_origin,
                atom_mask_origin,
                atom_neighbor_list_changed,
                bond_neighbor_list_changed,
                frag_mask1,
                frag_mask2,
                bond_index,
                JT_bond_features,
                JT_atom_neighbor_list,
                JT_bond_neighbor_list,
                JT_mask]





class MolPredAttentiveFP(nn.Module):
    def __init__(
            self,
            atom_feature_size,
            bond_feature_size,
            FP_size,
            atom_layers,
            mol_layers,
            DNN_layers,
            output_size,
            drop_rate,
    ):
        super(MolPredAttentiveFP, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size = atom_feature_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )
        self.MolEmbedding = AttentiveFP_mol(
            layers = mol_layers,
            FP_size = FP_size,
            droprate = drop_rate
        )
        self.Classifier = DNN(
            input_size = FP_size,
            output_size = output_size,
            layer_sizes = DNN_layers,
        )

    def forward(self, Input):
        [atom_features, bond_features, atom_neighbor_list, bond_neighbor_list, atom_mask] = Input
        atom_features = atom_features.cuda()
        bond_features = bond_features.cuda()
        atom_neighbor_list = atom_neighbor_list.cuda()
        bond_neighbor_list = bond_neighbor_list.cuda()
        atom_mask = atom_mask.cuda()

        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list, bond_neighbor_list)
        mol_FP, _ = self.MolEmbedding(atom_FP, atom_mask)
        prediction = self.Classifier(mol_FP)
        return prediction
