import torch as t
from torch.utils import data
from FraGAT.ChemUtils import *
import rdkit.Chem as Chem
import re
import random
from FraGAT.Featurizer import *
from FraGAT.Transformer import *
from FraGAT.Splitter import *
from FraGAT.Checker import *

class FileLoader(object):
    def __init__(self, file_path):
        super(FileLoader, self).__init__()
        self.path = file_path

    def load(self):
        with open(self.path, 'r') as f:
            raw_data = f.readlines()
        return raw_data


class BasicFileParser(object):
    def __init__(self):
        super(BasicFileParser, self).__init__()

    def _parse_line(self, line):
        raise NotImplementedError(
            "Line parser not implemented.")

    def parse_file(self, raw_data):
        # This function is used to return a Dataset parsed from the given dataset file.
        # Dataset is a list of dicts in type of {'SMILES':'xxx', 'Value':value}
        # value is a list of strings, like ['1','0','1',...]
        Dataset = []
        for line in raw_data:
            data = self._parse_line(line)
            Dataset.append(data)
        return Dataset

class HIVFileParser(BasicFileParser):
    def __init__(self):
        super(HIVFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n',Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class BBBPFileParser(BasicFileParser):
    def __init__(self):
        super(BBBPFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n',Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class BACEFileParser(BasicFileParser):
    def __init__(self):
        super(BACEFileParser,self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n',Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class QM9FileParser(BasicFileParser):
    def __init__(self):
        super(QM9FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}

class FreeSolvFileParser(BasicFileParser):
    def __init__(self):
        super(FreeSolvFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class LipopFileParser(BasicFileParser):
    def __init__(self):
        super(LipopFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class MalariaFileParser(BasicFileParser):
    def __init__(self):
        super(MalariaFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}
    
class CEPFileParser(BasicFileParser):
    def __init__(self):
        super(CEPFileParser, self).__init__()
        
    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class SHP2FileParser(BasicFileParser):
    def __init__(self):
        super(SHP2FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class Tox21FileParser(BasicFileParser):
    def __init__(self):
        super(Tox21FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
        return {'SMILES': SMILES, 'Value': Value}

class ToxcastFileParser(BasicFileParser):
    def __init__(self):
        super(ToxcastFileParser, self).__init__()

    def _parse_line(self, line):
        # Convert '1.0/0.0' to '1/0'
        # Convert missing value '' to '-1'
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
            elif value == '0.0':
                Value[i] = '0'
            elif value == '1.0':
                Value[i] = '1'
        return {'SMILES': SMILES, 'Value': Value}

class MUVFileParser(BasicFileParser):
    def __init__(self):
        super(MUVFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
        return {"SMILES": SMILES, 'Value': Value}

class ClinToxFileParser(BasicFileParser):
    def __init__(self):
        super(ClinToxFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}

class SIDERFileParser(BasicFileParser):
    def __init__(self):
        super(SIDERFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}

class ESOLFileParser(BasicFileParser):
    def __init__(self):
        super(ESOLFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}
##########################################################################
class MolDatasetEval(data.Dataset):
    def __init__(self, dataset, opt):
        super(MolDatasetEval, self).__init__()
        self.dataset = dataset
        self.Frag = opt.args['Frag']
        self.opt = opt
        self.FeaturizerList = {
            'FP': FPFeaturizer(opt),
            'Graph': GraphFeaturizer(),
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=opt.args['atom_feature_size'],
                bond_feature_size=opt.args['bond_feature_size'],
                max_degree = 5,
                max_frag = 2,
                mode='EVAL'
            )
            #'SMILES': SMILESFeaturizer()
        }
        self.featurizer = self.FeaturizerList[opt.args['Feature']]

        # if use methods in AttentiveFP to construct dataset, some more works should be down here.
        if opt.args['Feature'] == 'AttentiveFP':
            print("Using Attentive FP. Dataset is being checked.")
            self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
            self.dataset = self.checker.check(self.dataset)       # screen invalid molecules
            print("Prefeaturizing molecules......")
            self.featurizer.GetPad(self.dataset)
            self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
            print("Prefeaturization complete.")

    def __getitem__(self, index):
        if self.featurizer.__class__ == AttentiveFPFeaturizer:
            value = self.dataset[index]["Value"]
            smiles = self.dataset[index]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            #print("Single bonds num: ", len(GetSingleBonds(mol)))
            data, label = self.featurizer.featurizenew(self.prefeaturized_dataset, index, mol, value, self.Frag, self.opt)
        else:
            item = self.dataset[index]
            data, label = self.featurizer.featurize(item)
        return data, label

    def __len__(self):
        return len(self.dataset)

class MolDatasetTrain(data.Dataset):
    def __init__(self, dataset, opt):
        super(MolDatasetTrain, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.Frag = self.opt.args['Frag']
        self.FeaturizerList = {
            'FP': FPFeaturizer(opt),
            'Graph': GraphFeaturizer(),
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=opt.args['atom_feature_size'],
                bond_feature_size=opt.args['bond_feature_size'],
                max_degree = 5,
                max_frag = 2,
                mode='TRAIN'
            )
            #'SMILES': SMILESFeaturizer()
        }
        self.featurizer = self.FeaturizerList[opt.args['Feature']]
        if 'max_atom_num' in self.opt.args:
            self.max_atom_num = self.opt.args['max_atom_num']
        else:
            self.max_atom_num = 102

        # if use methods in AttentiveFP to construct dataset, some more works should be down here.
        if opt.args['Feature'] == 'AttentiveFP':
            print("Using Attentive FP. Dataset is being checked.")
            self.checker = AttentiveFPChecker(max_atom_num=self.max_atom_num, max_degree=5)
            self.dataset = self.checker.check(self.dataset)       # screen invalid molecules
            print("Prefeaturizing molecules......")
            self.featurizer.GetPad(self.dataset)
            self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
            print("Prefeaturization complete.")

    def __getitem__(self, index):
        if self.featurizer.__class__ == AttentiveFPFeaturizer:
            value = self.dataset[index]["Value"]
            smiles = self.dataset[index]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            data, label = self.featurizer.featurizenew(self.prefeaturized_dataset, index, mol, value, self.Frag, self.opt)
        else:
            item = self.dataset[index]
            data, label = self.featurizer.featurize(item)
        return data, label

    def __len__(self):
        return len(self.dataset)
#########################################################################

##########################################################################

class MolDatasetCreator(object):
    # An object to create molecule datasets from a given dataset file path.
    # Using CreateDatasets function to generate 2 or 3 datasets, based on the SplitRate
    def __init__(self, opt):
        super(MolDatasetCreator, self).__init__()

        self.FileParserList = {
            'HIV': HIVFileParser(),
            'BBBP': BBBPFileParser(),
            'Tox21': Tox21FileParser(),
            'FreeSolv': FreeSolvFileParser(),
            'ESOL': ESOLFileParser(),
            'QM9': QM9FileParser(),
            'BACE': BACEFileParser(),
            'ClinTox': ClinToxFileParser(),
            'SIDER': SIDERFileParser(),
            'SHP2': SHP2FileParser(),
            'Toxcast': ToxcastFileParser(),
            'Toxcast1': ToxcastFileParser(),
            'Toxcast2': ToxcastFileParser(),
            'Toxcast3': ToxcastFileParser(),
            'Toxcast4': ToxcastFileParser(),
            'Lipop': LipopFileParser(),
            'CEP': CEPFileParser(),
            'CEP2': CEPFileParser(),
            'Malaria': MalariaFileParser(),
            'Malaria2': MalariaFileParser(),
            'MUV': MUVFileParser()
        }
        self.SplitterList = {
            'Random': RandomSplitter(),
            'MultitaskRandom': MultitaskRandomSplitter(),
            'Scaffold': ScaffoldSplitter(),
            'ScaffoldRandom': ScaffoldRandomSplitter()
        }
        self.TransformerList = {
            'Augmentation': BinaryClassificationAugmentationTransformer()
        }

        self.opt = opt

    def CreateDatasets(self):
        file_path = self.opt.args['DataPath']
        print("Loading data file...")
        fileloader = FileLoader(file_path)
        raw_data = fileloader.load()

        print("Parsing lines...")
        parser = self.FileParserList[self.opt.args['ExpName']]
        raw_dataset = parser.parse_file(raw_data)
        # raw_dataset is a list in type of : {'SMILES': , 'Value': }
        print("Dataset is parsed. Original size is ", len(raw_dataset))

        if self.opt.args['ClassNum'] == 2:         # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(raw_dataset)
            else:
                weights = None
        else:
            weights = None
        # weights is a list with length of 'TaskNum'.
        # It shows the distribution of pos/neg samples in the dataset(Original dataset, before splitting)
        # And it is used as a parameter of the loss function to balance the learning process.
        # For multitasks, it contains multiple weights.


        if self.opt.args['Splitter']:
            if (self.opt.args['Splitter'] == 'Scaffold') or (self.opt.args['Splitter'] == 'ScaffoldRandom'):
                checker = ScaffoldSplitterChecker()
                raw_dataset = checker.check(raw_dataset)
                # if use scaffold splitter, the invalid smiles should be discarded from raw_dataset first.

            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets = splitter.split(raw_dataset, self.opt)
            if len(sets) == 2:
                trainset, validset = sets
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))
            if len(sets) == 3:
                trainset, validset, testset = sets
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset), " and testset: ", len(testset))
        else:
            trainset = raw_dataset
            sets = (trainset)

        # if the weight is used, then the augmentation should not be used.
        if self.opt.args['ClassNum'] == 2:
            if not self.opt.args['Weight']:
                if self.opt.args['Augmentation']:
                    transformer = self.TransformerList['Augmentation']
                    print("Augmentating...")
                    trainset = transformer.transform(trainset)
                    print("Trainset is enlarged to size: ", len(trainset))
                    if self.opt.args['ValidBalance']:
                        validset = transformer.transform(validset)
                        print("Validset is enlarged to size: ", len(validset))
                    if self.opt.args['TestBalance']:
                        testset = transformer.transform(testset)
                        print("Testset is enlarged to size: ", len(testset))

        # Construct sub datasets.
        Trainset = MolDatasetTrain(trainset, self.opt)
        if len(sets) == 2:
            Validset = MolDatasetEval(validset, self.opt)
            return (Trainset, Validset), weights
        elif len(sets) == 3:
            Validset = MolDatasetEval(validset, self.opt)
            Testset = MolDatasetEval(testset, self.opt)
            return (Trainset, Validset, Testset), weights
        else:
            return (Trainset), weights

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == '0':
                    neg_count += 1
                elif value == '1':
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights
