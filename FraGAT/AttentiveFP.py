import torch as t
import torch.nn as nn
import torch.nn.functional as F
import rdkit
import rdkit.Chem as Chem

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x

class AttentionCalculator(nn.Module):
    def __init__(self, FP_size, droprate):
        super(AttentionCalculator, self).__init__()
        self.FP_size = FP_size
        #self.align = nn.Linear(2*self.FP_size, 1)
        self.align = LinearBn(2*self.FP_size, 1)
        self.dropout = nn.Dropout(p = droprate)

    def forward(self, FP_align, atom_neighbor_list):
        # size of input Tensors:
        # FP_align: [batch_size, max_atom_length, max_neighbor_length, 2*FP_size]
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length]

        batch_size, max_atom_length, max_neighbor_length, _ = FP_align.size()

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_neighbor_list.clone()
        attend_mask[attend_mask != max_atom_length - 1] = 1
        attend_mask[attend_mask == max_atom_length - 1] = 0
        attend_mask = attend_mask.type(t.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_neighbor_list.clone()
        softmax_mask[softmax_mask != max_atom_length - 1] = 0
        softmax_mask[softmax_mask == max_atom_length - 1] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(t.cuda.FloatTensor).unsqueeze(-1)

        # size of masks: [batch_szie, max_atom_length, max_neighbor_length, 1]

        # calculate attention value
        align_score = self.align(self.dropout(FP_align))
        align_score = F.leaky_relu(align_score)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, dim=-2)
        attention_weight = attention_weight * attend_mask
        # size: [batch_size, max_atom_length, max_neighbor_length, 1]

        return attention_weight

class ContextCalculator(nn.Module):
    def __init__(self, FP_size, droprate):
        super(ContextCalculator, self).__init__()
        #self.attend = nn.Linear(FP_size, FP_size)
        self.attend = LinearBn(FP_size, FP_size)
        self.dropout = nn.Dropout(p = droprate)

    def forward(self, neighbor_FP, attention_score):
        # size of input Tensors:
        # neighbor_FP: [batch_size, max_atom_length, max_neighbor_length, FP_size]
        # attention_score: [batch_size, max_atom_length, max_neighbor_length, 1]

        neighbor_FP = self.dropout(neighbor_FP)
        neighbor_FP = self.attend(neighbor_FP)
        context = t.sum(t.mul(attention_score, neighbor_FP), -2)    #after sum, the dim -2 disappears.
        context = F.elu(context)
        # context size: [batch_size, max_atom_length, FP_size]
        # a context vector for each atom in each molecule.
        return context

class FPTranser(nn.Module):
    def __init__(self, FP_size):
        super(FPTranser, self).__init__()
        self.FP_size = FP_size
        self.GRUCell = nn.GRUCell(self.FP_size, self.FP_size)

    def forward(self, atom_FP, context_FP, atom_neighbor_list):
        # size of input Tensors:
        # atom_FP: [batch_size, max_atom_length, FP_size]
        # context_FP: [batch_size, max_atom_length, FP_size]

        batch_size, max_atom_length, _ = atom_FP.size()

        # GRUCell cannot treat 3D Tensors.
        # flat the mol dim and atom dim.
        context_FP_reshape = context_FP.view(batch_size * max_atom_length, self.FP_size)
        atom_FP_reshape = atom_FP.view(batch_size * max_atom_length, self.FP_size)
        new_atom_FP_reshape = self.GRUCell(context_FP_reshape, atom_FP_reshape)
        new_atom_FP = new_atom_FP_reshape.view(batch_size, max_atom_length, self.FP_size)
        activated_new_atom_FP = F.relu(new_atom_FP)
        # size:
        # [batch_size, max_atom_length, FP_size]

        # calculate new_neighbor_FP
        new_neighbor_FP = [activated_new_atom_FP[i][atom_neighbor_list[i]] for i in range(batch_size)]
        new_neighbor_FP = t.stack(new_neighbor_FP, dim=0)
        # size:
        # [batch_size, max_atom_length, max_neighbor_length, FP_size]

        return new_atom_FP, activated_new_atom_FP, new_neighbor_FP

class FPInitializer(nn.Module):
    def __init__(self, atom_feature_size, bond_feature_size, FP_size):
        super(FPInitializer, self).__init__()
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.FP_size = FP_size
        #self.atom_fc = nn.Linear(self.atom_feature_size, self.FP_size)
        self.atom_fc = LinearBn(self.atom_feature_size, self.FP_size)
        #self.nei_fc = nn.Linear((self.atom_feature_size + self.bond_feature_size), self.FP_size)
        self.nei_fc = LinearBn((self.atom_feature_size + self.bond_feature_size), self.FP_size)

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        batch_size, max_atom_length, _ = atom_features.size()

        # generate atom_neighbor_features from atom_features by atom_neighbor_list, size is [batch, atom, neighbor, atom_feature_size]
        atom_neighbor_features = [atom_features[i][atom_neighbor_list[i]] for i in range(batch_size)]
        atom_neighbor_features = t.stack(atom_neighbor_features, dim=0)

        # generate bond_neighbor_features from bond_features by bond_neighbor list, size is [batch, atom, neighbor, bond_feature_size]
        bond_neighbor_features = [bond_features[i][bond_neighbor_list[i]] for i in range(batch_size)]
        bond_neighbor_features = t.stack(bond_neighbor_features, dim=0)

        # concate bond_neighbor_features and atom_neighbor_features, and then transform it from
        # [batch, atom, neighbor, atom_feature_size+bond_feature_size] to [batch, atom, neighbor, FP_size]
        neighbor_FP = t.cat([atom_neighbor_features, bond_neighbor_features], dim=-1)
        #print(neighbor_FP.size())
        neighbor_FP = self.nei_fc(neighbor_FP)
        neighbor_FP = F.leaky_relu(neighbor_FP)

        # transform atom_features from [batch, atom, atom_feature_size] to [batch, atom, FP_size]
        atom_FP = self.atom_fc(atom_features)
        atom_FP = F.leaky_relu(atom_FP)

        return atom_FP, neighbor_FP

class FPInitializerNew(nn.Module):
    # with mixture item.
    def __init__(self, atom_feature_size, bond_feature_size, FP_size, droprate):
        super(FPInitializerNew, self).__init__()
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.FP_size = FP_size
        #self.atom_fc = nn.Linear(self.atom_feature_size, self.FP_size)
        self.atom_fc = nn.Sequential(
            LinearBn(self.atom_feature_size, self.FP_size),
            nn.ReLU(inplace = True),
            nn.Dropout(p = droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace = True)
        )
        self.bond_fc = nn.Sequential(
            LinearBn(self.bond_feature_size, self.FP_size),
            nn.ReLU(inplace = True),
            nn.Dropout(p=droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace = True)
        )
        self.nei_fc = nn.Sequential(
            LinearBn(3*self.FP_size, self.FP_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace = True)
        )
        #self.nei_fc = nn.Linear((self.atom_feature_size + self.bond_feature_size), self.FP_size)
        #self.nei_fc = LinearBn((self.atom_feature_size + self.bond_feature_size), self.FP_size)

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        batch_size, max_atom_length, _ = atom_features.size()

        atom_FP = self.atom_fc(atom_features)
        #[batch_size, max_atom_length, FP_size]

        bond_FP = self.bond_fc(bond_features)
        #[batch_size, max_bond_length, FP_size]

        # generate atom_neighbor_FP from atom_FP by atom_neighbor_list,
        # size is [batch, atom, neighbor, FP_size]
        atom_neighbor_FP = [atom_FP[i][atom_neighbor_list[i]] for i in range(batch_size)]
        atom_neighbor_FP = t.stack(atom_neighbor_FP, dim=0)

        # generate bond_neighbor_FP from bond_FP by bond_neighbor list,
        # size is [batch, atom, neighbor, FP_size]
        bond_neighbor_FP = [bond_FP[i][bond_neighbor_list[i]] for i in range(batch_size)]
        bond_neighbor_FP = t.stack(bond_neighbor_FP, dim=0)

        # generate mixture item
        # size is [batch, atom, neighbor, FP_size]
        mixture = atom_neighbor_FP + bond_neighbor_FP - atom_neighbor_FP * bond_neighbor_FP

        # concate bond_neighbor_FP and atom_neighbor_FP and mixture item, and then transform it from
        # [batch, atom, neighbor, 3*FP_size] to [batch, atom, neighbor, FP_size]
        neighbor_FP = t.cat([atom_neighbor_FP, bond_neighbor_FP, mixture], dim=-1)
        #print(neighbor_FP.size())
        neighbor_FP = self.nei_fc(neighbor_FP)
        #neighbor_FP = F.leaky_relu(neighbor_FP)

        # transform atom_features from [batch, atom, atom_feature_size] to [batch, atom, FP_size]
        #atom_FP = self.atom_fc(atom_features)
        #atom_FP = F.leaky_relu(atom_FP)

        return atom_FP, neighbor_FP
###########################################################################################################
class AttentiveFPLayer(nn.Module):
    def __init__(self, FP_size, droprate):
        super(AttentiveFPLayer, self).__init__()
        self.FP_size = FP_size
        self.attentioncalculator = AttentionCalculator(self.FP_size, droprate)
        self.contextcalculator = ContextCalculator(self.FP_size, droprate)
        self.FPtranser = FPTranser(self.FP_size)

    def forward(self, atom_FP, neighbor_FP, atom_neighbor_list):
        # align atom FP and its neighbors' FP to generate [hv, hu]
        FP_align = self.feature_align(atom_FP, neighbor_FP)
        # FP_align: [batch_size, max_atom_length, max_neighbor_length, 2*FP_size]

        # calculate attention score evu
        attention_score = self.attentioncalculator(FP_align, atom_neighbor_list)
        # attention_score: [batch_size, max_atom_length, max_neighbor_length, 1]

        # calculate context FP
        context_FP = self.contextcalculator(neighbor_FP, attention_score)
        # context_FP: [batch_size, max_atom_length, FP_size)

        # transmit FPs between atoms.
        activated_new_atom_FP, new_atom_FP, neighbor_FP = self.FPtranser(atom_FP, context_FP, atom_neighbor_list)

        return activated_new_atom_FP, new_atom_FP, neighbor_FP

    def feature_align(self, atom_FP, neighbor_FP):
        # size of input Tensors:
        # atom_FP: [batch_size, max_atom_length, FP_size]
        # neighbor_FP: [batch_size, max_atom_length, max_neighbor_length, FP_size]

        batch_size, max_atom_length, max_neighbor_length, _ = neighbor_FP.size()

        atom_FP = atom_FP.unsqueeze(-2)
        # [batch_size, max_atom_length, 1, FP_size]
        atom_FP = atom_FP.expand(batch_size, max_atom_length, max_neighbor_length, self.FP_size)
        # [batch_size, max_atom_length, max_neighbor_length, FP_size]

        FP_align = t.cat([atom_FP, neighbor_FP], dim=-1)
        # size: [batch_size, max_atom_length, max_neighborlength, 2*FP_size]

        return FP_align


###########################################################################################################
class AttentiveFP_atom(nn.Module):
    def __init__(self, atom_feature_size, bond_feature_size, FP_size, layers, droprate):
        super(AttentiveFP_atom, self).__init__()
        self.FPinitializer = FPInitializerNew(atom_feature_size, bond_feature_size, FP_size, droprate)
        self.AttentiveFPLayers = nn.ModuleList()
        for i in range(layers):
            self.AttentiveFPLayers.append(AttentiveFPLayer(FP_size, droprate))

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        # atom_features and neighbor_features initializing.
        atom_FP, neighbor_FP = self.FPinitializer(atom_features, bond_features, atom_neighbor_list, bond_neighbor_list)

        # use attentiveFPlayers to update the atom_features and the neighbor_features.
        for layer in self.AttentiveFPLayers:
            atom_FP, _, neighbor_FP = layer(atom_FP, neighbor_FP, atom_neighbor_list)

        return atom_FP

class AttentiveFP_mol(nn.Module):
    def __init__(self, layers, FP_size, droprate):
        super(AttentiveFP_mol, self).__init__()
        self.layers = layers
        self.FP_size = FP_size
        #self.align = nn.Linear(2 * self.FP_size, 1)
        self.align = LinearBn(2 * self.FP_size, 1)
        self.dropout = nn.Dropout(p = droprate)
        #self.attend = nn.Linear(self.FP_size, self.FP_size)
        self.attend = LinearBn(self.FP_size, self.FP_size)
        self.mol_GRUCell = nn.GRUCell(self.FP_size, self.FP_size)

    def forward(self, atom_FP, atom_mask):
        # size of input Tensors:
        # atom_FP: [batch_size, atom_length, FP_size]
        # atom_mask: [batch_size, atom_length]

        batch_size, max_atom_length, _ = atom_FP.size()
        atom_mask = atom_mask.unsqueeze(2)
        # [batch_size, atom_length, 1]
        super_node_FP = t.sum(atom_FP * atom_mask, dim=-2)
        # FP of super node S is set to be sum of all of the atoms' FP initially.
        # pad nodes is eliminated after atom_FP * atom_mask
        # super node FP size: [batch, FP_size]

        # generate masks to eliminate the influence of pad nodes.
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(t.cuda.FloatTensor)
        # [batch, atom_length, 1]

        activated_super_node_FP = F.relu(super_node_FP)

        for i in range(self.layers):
            super_node_FP_expand = activated_super_node_FP.unsqueeze(-2)
            super_node_FP_expand = super_node_FP_expand.expand(batch_size, max_atom_length, self.FP_size)
            # [batch, max_atom_length, FP_size]

            super_node_align = t.cat([super_node_FP_expand, atom_FP], dim=-1)
            # [batch, max_atom_length, 2*FP_size]
            super_node_align_score = self.align(super_node_align)
            super_node_align_score = F.leaky_relu(super_node_align_score)
            # [batch, max_atom_length, 1]
            super_node_align_score = super_node_align_score + mol_softmax_mask
            super_node_attention_weight = F.softmax(super_node_align_score, -2)
            super_node_attention_weight = super_node_attention_weight * atom_mask
            # [batch_size, max_atom_length, 1]

            atom_FP_transform = self.attend(self.dropout(atom_FP))
            # [batch_size, max_atom_length, FP_size]
            super_node_context = t.sum(t.mul(super_node_attention_weight, atom_FP_transform), -2)
            super_node_context = F.elu(super_node_context)
            # [batch_size, FP_size]
            # the dim -2 is eliminated after sum function.
            super_node_FP = self.mol_GRUCell(super_node_context, super_node_FP)

            # do nonlinearity
            activated_super_node_FP = F.relu(super_node_FP)

        return super_node_FP, activated_super_node_FP



        ###########################################################

#############################################################################################################
# Test codes
'''
model1 = AttentiveFP_atom(atom_feature_size = 20, bond_feature_size = 5, FP_size = 150, layers = 3, droprate = 0.5)
model1.cuda()
print(model1)

for name, param in model1.named_parameters():
    print(name)
    print(param)

batch_size = 1
max_atom_size = 12
max_bond_size = 12
atom_feature_size = 20
bond_feature_size = 5


atom_neighbor_list = [[
    [1,11,11,11,11],
    [0,2,11,11,11],
    [1,3,11,11,11],
    [2,4,7,11,11],
    [3,5,11,11,11],
    [4,6,11,11,11],
    [5,8,10,11,11],
    [3,8,9,11,11],
    [6,7,11,11,11],
    [7,11,11,11,11],
    [6,11,11,11,11],
    [11,11,11,11,11]
]]
atom_neighbor_list = t.Tensor(atom_neighbor_list)
print(atom_neighbor_list.size())

bond_neighbor_list = [[
    [0,11,11,11,11],
    [0,1,11,11,11],
    [1,2,11,11,11],
    [2,3,9,11,11],
    [3,4,11,11,11],
    [4,5,11,11,11],
    [5,6,7,11,11],
    [8,9,11,11,11],
    [7,8,11,11,11],
    [10,11,11,11,11],
    [6,11,11,11,11],
    [11,11,11,11,11]
]]
bond_neighbor_list = t.Tensor(bond_neighbor_list)
print(bond_neighbor_list.size())

atom_feature = t.Tensor(batch_size, max_atom_size, atom_feature_size)
bond_feature = t.Tensor(batch_size, max_bond_size, bond_feature_size)
atom_feature[0,11,:]=0
bond_feature[0,11,:]=0
print(atom_feature.size())
print(bond_feature.size())

atom_mask = [[1,1,1,1,1,1,1,1,1,1,1,0]]
atom_mask = t.Tensor(atom_mask)
print(atom_mask.size())

atom_feature = atom_feature.cuda()
bond_feature = bond_feature.cuda()
atom_neighbor_list = atom_neighbor_list.long().cuda()
bond_neighbor_list = bond_neighbor_list.long().cuda()
output = model1(atom_feature, bond_feature, atom_neighbor_list, bond_neighbor_list)

print(output.size())
print(output)

model2 = AttentiveFP_mol(layers = 3, droprate = 0.5, FP_size = 150)
model2.cuda()
print(model2)
for name, params in model2.named_parameters():
    print(name)
    print(params)

atom_mask = atom_mask.cuda()
_, MolEmbedding = model2(output, atom_mask)

print(MolEmbedding.size())
print(MolEmbedding)

'''