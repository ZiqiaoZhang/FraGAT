import random
import numpy as np
from FraGAT.ChemUtils import *





class BasicSplitter(object):
    def __init__(self):
        super(BasicSplitter, self).__init__()


    def split(self, dataset, opt):
        raise NotImplementedError(
            "Dataset Splitter not implemented.")

class RandomSplitter(BasicSplitter):
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def CheckClass(self, dataset, tasknum):
        c0cnt = np.zeros(tasknum)
        c1cnt = np.zeros(tasknum)
        for data in dataset:
            value = data['Value']
            assert tasknum == len(value)
            for task in range(tasknum):
                if value[task] == '0':
                    c0cnt[task]+=1
                elif value[task] == '1':
                    c1cnt[task]+=1
        if 0 in c0cnt:
            print("Invalid splitting.")
            return False
        elif 0 in c1cnt:
            print("Invalid splitting.")
            return False
        else:
            return True

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        if len(rate) == 1:
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(dataset)
                set1 = dataset[:train_num]
                set2 = dataset[train_num:]

                assert len(set1) == train_num
                assert len(set2) == valid_num
                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1
            return (set1, set2)

        if len(rate) == 2:
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
            endflag = 0
            while not endflag:
                random.seed(testseed)
                random.shuffle(dataset)
                set3 = dataset[(train_num + valid_num):]
                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set3, opt.args['TaskNum'])
                    testseed += 1
                else:
                    endflag = 1

            set_remain = dataset[:(train_num + valid_num)]
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(set_remain)
                set1 = set_remain[:train_num]
                set2 = set_remain[train_num:]

                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1

                assert len(set1) == train_num
                assert len(set2) == valid_num
                assert len(set3) == test_num

            return (set1,set2,set3)

class MultitaskRandomSplitter(BasicSplitter):
    def __init__(self):
        super(MultitaskRandomSplitter, self).__init__()

    def BuildEntireDataset(self, dataset):
        EntireDataset = {'Index':[], 'Items': []}
        for i in range(len(dataset)):
            EntireDataset['Index'].append(i)
            EntireDataset['Items'].append(dataset[i])
        return EntireDataset

    def merge(self, EntireDataset, EntireSet, set2_index, set2_value, task, task_num):
        # entire_set: {'Index': [], 'Items': [{'SMILES':, 'Value':}]}
        # set2_index: list of index of the set to be merged.
        # set2_value: value of the set to be merged
        NewSet = EntireSet.copy()
        EntireSetIndex = EntireSet['Index']
        for j in range(len(set2_index)):
            item_ind = set2_index[j]
            item_value = set2_value[j]
            if item_ind in EntireSetIndex:
                loc = EntireSetIndex.index(item_ind)
                NewSet['Items'][loc]['Value'][task] = item_value
            else:
                NewSet['Index'].append(item_ind)
                SMILES = EntireDataset['Items'][item_ind]['SMILES']
                Value = []
                for k in range(task_num):
                    Value.append(-1)
                Value[task] = item_value
                NewSet['Items'].append({'SMILES': SMILES, 'Value':Value})
        return NewSet


    def OneTaskSplit(self, EntireDataset, task, rate, validseed=0, testseed=0):
        print('Splitting task', task)
        EntireNum = len(EntireDataset['Index'])
        TaskIndexPosSet = []
        TaskIndexNegSet = []
        #print(EntireDataset)
        for i in range(EntireNum):
            data = EntireDataset['Items'][i]
            if data['Value'][task] == '0':
                TaskIndexNegSet.append(i)
            elif data['Value'][task] == '1':
                TaskIndexPosSet.append(i)
        TaskPosNum = len(TaskIndexPosSet)
        TaskNegNum = len(TaskIndexNegSet)
        print("TaskPosNum:", TaskPosNum)
        print("TaskNegNum:", TaskNegNum)

        if len(rate) == 1:
            TaskPosTrainNum = int(TaskPosNum * rate[0])
            TaskPosValidNum = TaskPosNum - TaskPosTrainNum
            TaskNegTrainNum = int(TaskNegNum * rate[0])
            TaskNegValidNum = TaskNegNum - TaskNegTrainNum
            assert TaskPosValidNum > 0
            assert TaskPosTrainNum > 0
            assert TaskNegValidNum > 0
            assert TaskNegTrainNum > 0

            random.seed(validseed)
            random.shuffle(TaskIndexPosSet)
            random.shuffle(TaskIndexNegSet)
            TaskPosTrainSet = TaskIndexPosSet[:TaskPosTrainNum]
            TaskPosValidSet = TaskIndexPosSet[TaskPosTrainNum:]
            TaskNegTrainSet = TaskIndexNegSet[:TaskNegTrainNum]
            TaskNegValidSet = TaskIndexNegSet[TaskNegTrainNum:]

            TaskTrainSet = TaskPosTrainSet + TaskNegTrainSet
            TaskTrainValueSet = []
            for i in range(len(TaskPosTrainSet)):
                TaskTrainValueSet.append(1)
            for i in range(len(TaskNegTrainSet)):
                TaskTrainValueSet.append(0)
            TaskValidSet = TaskPosValidSet + TaskNegValidSet
            TaskValidValueSet = []
            for i in range(len(TaskPosValidSet)):
                TaskValidValueSet.append(1)
            for i in range(len(TaskNegValidSet)):
                TaskValidValueSet.append(0)

            assert len(TaskTrainSet) == TaskPosTrainNum + TaskNegTrainNum
            assert len(TaskValidSet) == TaskPosValidNum + TaskNegValidNum

            return (TaskTrainSet, TaskValidSet), (TaskTrainValueSet, TaskValidValueSet)

        elif len(rate) == 2:
            TaskPosTrainNum = int(TaskPosNum * rate[0])
            TaskPosValidNum = int(TaskPosNum * rate[1])
            TaskPosTestNum = TaskPosNum - TaskPosTrainNum - TaskPosValidNum
            TaskNegTrainNum = int(TaskNegNum * rate[0])
            TaskNegValidNum = int(TaskNegNum * rate[1])
            TaskNegTestNum = TaskNegNum - TaskNegTrainNum - TaskNegValidNum

            assert TaskPosTrainNum > 0
            assert TaskPosValidNum > 0
            assert TaskPosTestNum > 0
            assert TaskNegTrainNum > 0
            assert TaskNegValidNum > 0
            assert TaskNegTestNum > 0

            random.seed(testseed)
            random.shuffle(TaskIndexPosSet)
            random.shuffle(TaskIndexNegSet)
            TaskPosRemainSet = TaskIndexPosSet[:(TaskPosTrainNum + TaskPosValidNum)]
            TaskPosTestSet = TaskIndexPosSet[(TaskPosTrainNum + TaskPosValidNum):]
            TaskNegRemainSet = TaskIndexNegSet[:(TaskNegTrainNum + TaskNegValidNum)]
            TaskNegTestSet = TaskIndexNegSet[(TaskNegTrainNum + TaskNegValidNum):]

            random.seed(validseed)
            random.shuffle(TaskPosRemainSet)
            random.shuffle(TaskNegRemainSet)
            TaskPosTrainSet = TaskPosRemainSet[:TaskPosTrainNum]
            TaskPosValidSet = TaskPosRemainSet[TaskPosTrainNum:]
            TaskNegTrainSet = TaskNegRemainSet[:TaskNegTrainNum]
            TaskNegValidSet = TaskNegRemainSet[TaskNegTrainNum:]

            TaskTrainSet = TaskPosTrainSet + TaskNegTrainSet
            TaskTrainValueSet = []
            for i in range(len(TaskPosTrainSet)):
                TaskTrainValueSet.append(1)
            for i in range(len(TaskNegTrainSet)):
                TaskTrainValueSet.append(0)
            TaskValidSet = TaskPosValidSet + TaskNegValidSet
            TaskValidValueSet = []
            for i in range(len(TaskPosValidSet)):
                TaskValidValueSet.append(1)
            for i in range(len(TaskNegValidSet)):
                TaskValidValueSet.append(0)
            TaskTestSet = TaskPosTestSet + TaskNegTestSet
            TaskTestValueSet = []
            for i in range(len(TaskPosTestSet)):
                TaskTestValueSet.append(1)
            for i in range(len(TaskNegTestSet)):
                TaskTestValueSet.append(0)

            assert len(TaskTrainSet) == TaskPosTrainNum + TaskNegTrainNum
            assert len(TaskValidSet) == TaskPosValidNum + TaskNegValidNum
            assert len(TaskTestSet) == TaskPosTestNum + TaskNegTestNum

            return (TaskTrainSet, TaskValidSet, TaskTestSet), (TaskTrainValueSet, TaskValidValueSet, TaskTestValueSet)

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        task_num = opt.args['TaskNum']
        EntireDataset = self.BuildEntireDataset(dataset)

        EntireTrainSet = {'Index':[], 'Items':[]}
        EntireValidSet = {'Index':[], 'Items':[]}
        EntireTestSet = {'Index':[], 'Items':[]}

        for i in range(task_num):
            sets, values = self.OneTaskSplit(EntireDataset, task = i, rate = rate, validseed = validseed, testseed = testseed)
            if len(rate) == 1:
                task_train_set, task_valid_set = sets
                task_train_value, task_valid_value = values
                EntireTrainSet = self.merge(EntireDataset, EntireTrainSet, task_train_set, task_train_value, i, task_num)
                EntireValidSet = self.merge(EntireDataset, EntireValidSet, task_valid_set, task_valid_value, i, task_num)

            elif len(rate) == 2:
                task_train_set, task_valid_set, task_test_set = sets
                task_train_value, task_valid_value, task_test_value = values
                EntireTrainSet = self.merge(EntireDataset, EntireTrainSet, task_train_set, task_train_value, i, task_num)
                EntireValidSet = self.merge(EntireDataset, EntireValidSet, task_valid_set, task_valid_value, i, task_num)
                EntireTestSet = self.merge(EntireDataset, EntireTestSet, task_test_set, task_test_value, i, task_num)

        if len(rate)==1:
            return (EntireTrainSet['Items'], EntireValidSet['Items'])
        if len(rate)==2:
            return (EntireTrainSet['Items'], EntireValidSet['Items'], EntireTestSet['Items'])
        # for each task:
        # First, extract data with valid values on task_i
        # Second, divide extracted data into positive and negative sets
        # Third, split positive set and negative set into train/valid/test sets by rates
        # merge pos and neg into train_task, valid_task, test_task sets.
        # Merge train_task with entire_task, the same as valid and test.

        # To merge pos and neg sets, because they are not overlapping, so it is easy to merge.
        # To merge task set and the entire set:
        # In the entire set, if an item is added into this set, its smiles and index will be stored, and the Values are set to be a vector with task_num length and all of the values are -1.
        # Then, for task_i, the Value[i] of this item will set to be either 1 or 0.
        # And for task_j, when adding an item, first

'''
        if len(rate) == 1:
            entire_trainset = []
            entire_validset = []

            for i in range(task_num):
                task_total_index = []
                for j in range(total_num):
                    data = dataset[j].copy()
                    Values = data['Value']
                    value = Values[i]
                    if value != '-1':
                        task_total_index.append(j)
                task_total_num = len(task_total_index)
                task_train_num = int(task_total_num * rate[0])
                task_valid_num = task_total_num - task_train_num
                random.seed(validseed)
                random.shuffle(task_total_index)
                set1_index = task_total_index[:task_train_num]
                set2_index = task_total_index[task_train_num:]
                set1 = [dataset[x] for x in set1_index]
                set2 = [dataset[y] for y in set2_index]
                entire_trainset = list(set(entire_trainset + set1))
                entire_validset = list(set(entire_validset + set2))


        #if len(rate) == 2:
        '''


class ScaffoldSplitter(BasicSplitter):
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    def generate_scaffold(self, smiles, include_chirality = False):
        generator = ScaffoldGenerator(include_chirality = include_chirality)
        scaffold = generator.get_scaffold(smiles)
        return scaffold

    def id2data(self, dataset, ids):
        new_dataset = []
        for id in ids:
            data = dataset[id]
            new_dataset.append(data)
        return new_dataset

    def split(self, dataset, opt):
        rate = opt.args['Split_rate']
        total_num = len(dataset)
        scaffolds = {}

        for id, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)

            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}

        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        if len(rate) == 1:
            assert rate[0] < 1
            train_num = int(total_num * rate[0])
        elif len(rate) == 2:
            assert rate[0]+rate[1] < 1
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
        else:
            print("Wrong splitting rate")
            raise RuntimeError

        trainids = []
        validids = []
        testids = []

        for scaffold_set in scaffold_sets:
            if len(rate)==1:
                if len(trainids) + len(scaffold_set) > train_num:
                    validids += scaffold_set
                else:
                    trainids += scaffold_set
            else:
                if len(trainids) + len(scaffold_set) > train_num:
                    if len(validids) + len(scaffold_set) > valid_num:
                        testids += scaffold_set
                    else:
                        validids += scaffold_set
                else:
                    trainids += scaffold_set

        trainset = self.id2data(dataset, trainids)
        validset = self.id2data(dataset, validids)
        if len(rate)==2:
            testset = self.id2data(dataset, testids)
            return (trainset, validset, testset)
        else:
            return (trainset, validset)

class ScaffoldRandomSplitter(BasicSplitter):
    def __init__(self):
        super(ScaffoldRandomSplitter, self).__init__()

    def generate_scaffold(self, smiles, include_chirality = False):
        generator = ScaffoldGenerator(include_chirality = include_chirality)
        scaffold = generator.get_scaffold(smiles)
        return scaffold

    def id2data(self, dataset, ids):
        new_dataset = []
        for id in ids:
            data = dataset[id]
            new_dataset.append(data)
        return new_dataset

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        scaffolds = {}

        # extract scaffolds.
        for id, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)

            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        # scaffolds is a dict, scaffold is the key, and the index of the molecules that have the scaffold are the value.
        # {scaffold: [ind]}
        # for random selection, the scaffolds does not need to be sorted.

        # calculate splitting number
        if len(rate) == 1:
            assert rate[0] < 1
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
        elif len(rate) == 2:
            assert rate[0]+rate[1] < 1
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
        else:
            print("Wrong splitting rate")
            raise RuntimeError
        # optimal counts: test_num, valid_num, train_num

        # For different task# and class#, the processing method is different.
        # While in the current version, only the method for 1 task 2 classes has been implemented.
        tasknum = opt.args['TaskNum']
        classnum = opt.args['ClassNum']  # for regression task, classnum is sest to be 1
        if classnum == 2:   # case: binary classification
            if tasknum == 1:    # case: single task
                minorclass, minor_count, minor_ratio = self.BinarySingleTaskMinorClassCount(dataset)
            else:    # case: multi-task
                minorclass, minor_count, minor_ratio = self.BinaryMultiTaskMinorClassCount(dataset)
            # minorclass is an string of '0' or '1'
            # minor_count is the number of entire minor class samples in the dataset.
            # minor_ratio is the ratio of the minor class samples in the dataset.
            # minor_count is not useful.

            scaffold_keys = scaffolds.keys()  # sample scaffolds from scaffold_keys
            if len(rate) == 1:  # only sample the validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0]))
                # trick: the ratio of the sampled scaffolds is equal to the ratio of the sampled molecules.
                # which means that the program hope to sample the scaffold_sets in an average size.
                # So that the scaffold_set sampled is not too small.
                validids, _ = self.BinaryClassSample(dataset, scaffolds, sample_size, valid_num, minor_ratio, minorclass, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset)
            elif len(rate) == 2:  # sample testset then validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
                testids, chosen_scaffolds = self.BinaryClassSample(dataset, scaffolds, sample_size, test_num, minor_ratio,
                                                        minorclass, testseed)
                testset = self.id2data(dataset, testids)
                # remain_scaffolds = self.excludedscaffolds(scaffold_keys, chosen_scaffolds)
                remain_scaffolds = {x: scaffolds[x] for x in scaffolds.keys() if x not in chosen_scaffolds}
                sample_size = int(len(remain_scaffolds.keys()) * rate[1])
                validids, _ = self.BinaryClassSample(dataset, remain_scaffolds, sample_size, valid_num, minor_ratio, minorclass,
                                          validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids + testids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset, testset)

        elif classnum == 1: # case: regression
            scaffold_keys = scaffolds.keys()  # sample scaffolds from scaffold_keys
            if len(rate) == 1:  # only sample the validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0]))
                # trick: the ratio of the sampled scaffolds is equal to the ratio of the sampled molecules.
                # which means that the program hope to sample the scaffold_sets in an average size.
                # So that the scaffold_set sampled is not too small.
                validids, _ = self.RegressionSample(scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset)
            elif len(rate) == 2:  # sample testset then validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
                testids, chosen_scaffolds = self.RegressionSample(scaffolds, sample_size, test_num, testseed)
                testset = self.id2data(dataset, testids)
                # remain_scaffolds = self.excludedscaffolds(scaffold_keys, chosen_scaffolds)
                remain_scaffolds = {x: scaffolds[x] for x in scaffolds.keys() if x not in chosen_scaffolds}
                sample_size = int(len(remain_scaffolds.keys()) * rate[1])
                validids, _ = self.RegressionSample(remain_scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids + testids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset, testset)

        # random split


    #training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    #def excludedscaffolds(self, keys, chosen_scaffolds):
    #    remain_scaffolds = []
    #    for key in keys:
    #        if key not in chosen_scaffolds:
    #            remain_scaffolds.append(key)
    #    return remain_scaffolds
    def RegressionSample(self, scaffolds, sample_size, optimal_count, seed):
        count = 0
        keys = scaffolds.keys()
        tried_times = 0
        error_rate = 0.1
        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)):
            tried_times += 1
            if tried_times % 5000 == 0:
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample scaffold number.")
                sample_size = int(sample_size * 1.1)
                print(len(list(scaffolds.keys())))
                print(sample_size)
            seed += 1
            random.seed(seed)
            #print(len(list(keys)))
            #print(sample_size)
            chosen_scaffolds = random.sample(list(keys), sample_size)
            #print(chosen_scaffolds)
            count = sum([len(scaffolds[scaffold]) for scaffold in chosen_scaffolds])
            index = [index for scaffold in chosen_scaffolds for index in scaffolds[scaffold]]

        print("Sample num: ", count)
        print("Available Seed: ", seed)
        print("Tried times: ", tried_times)
        return index, chosen_scaffolds

    def BinaryClassSample(self, dataset, scaffolds, sample_size, optimal_count, minor_ratio, minor_class, seed):
        optimal_minor_count = minor_ratio * optimal_count
        count = 0
        minor_count = 0
        keys = scaffolds.keys()
        tried_times = 0
        failure_flag = 0
        error_rate = 0.1
        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)) \
            or (minor_count < optimal_minor_count * (1-error_rate)) or (minor_count > optimal_minor_count * (1+error_rate)):
            tried_times += 1
            # if cannot find one result:
            if tried_times % 5000 == 0:
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample scaffold number.")
                sample_size = int(sample_size * 1.1)
                print(len(list(scaffolds.keys())))
                print(sample_size)

                '''
                print("Randomly sample failure.")
                failure_flag = 1
                count = 0
                index = []
                scaffold_sets = [
                    scaffold_set for (scaffold, scaffold_set) in sorted(
                        scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
                    )
                ]
                for scaffold_set in scaffold_sets:
                    if count + len(scaffold_set) < optimal_count:
                        count += len(scaffold_set)
                        index += scaffold_set

                '''
            seed += 1
            random.seed(seed)
            #print(len(list(keys)))
            #print(sample_size)
            chosen_scaffolds = random.sample(list(keys), sample_size)
            #print(chosen_scaffolds)
            count = sum([len(scaffolds[scaffold]) for scaffold in chosen_scaffolds])
            index = [index for scaffold in chosen_scaffolds for index in scaffolds[scaffold]]
            minor_count = self.id2valuecount(dataset, index, minor_class)

        print("Sample num: ", count)
        print("Minor class num: ", minor_count)
        print("Available Seed: ", seed)
        print("Tried times: ", tried_times)
        return index, chosen_scaffolds

    def id2valuecount(self, dataset, ids, count_value):
        count = 0
        for id in ids:
            data = dataset[id]
            value = data['Value']
            if value == count_value:
                count += 1
        return count

    def excludedids(self, total_num, ids):
        excludedids = []
        ids.sort()
        j = 0
        for i in range(total_num):
            if j < len(ids):
                if i != ids[j]:
                    excludedids.append(i)
                else:
                    j += 1
            else:
                excludedids.append(i)
        assert len(excludedids) + len(ids) == total_num
        return excludedids

    def BinarySingleTaskMinorClassCount(self, dataset):
        class0_count = 0
        class1_count = 0
        for data in dataset:
            value = data['Value']
            if value == '0':
                class0_count += 1
            elif value == '1':
                class1_count += 1
            else:
                print("Value count error.")
                raise RuntimeError

        if class0_count > class1_count:
            minorclass = '1'
            minor_count = class1_count
            minor_ratio = class1_count / len(dataset)
        else:
            minorclass = '0'
            minor_count = class0_count
            minor_ratio = class0_count / len(dataset)

        return minorclass, minor_count, minor_ratio

    def BinaryMultiTaskMinorClassCount(self, dataset):
        class0_count = 0
        class1_count = 0
        for data in dataset:
            value = data['Value']
            if value[0] == '0':
                class0_count += 1
            elif value[0] == '1':
                class1_count += 1
            else:
                print("Value count error.")
                raise RuntimeError

        if class0_count > class1_count:
            minorclass = '1'
            minor_count = class1_count
            minor_ratio = class1_count / len(dataset)
        else:
            minorclass = '0'
            minor_count = class0_count
            minor_ratio = class0_count / len(dataset)

        return minorclass, minor_count, minor_ratio



