import random



class BasicTransformer(object):
    def __init__(self):
        super(BasicTransformer, self).__init__()

    def transform(self, dataset):
        raise NotImplementedError(
            "Dataset Transformer not implemented.")


class BinaryClassificationAugmentationTransformer(BasicTransformer):
    def __init__(self):
        super(BinaryClassificationAugmentationTransformer, self).__init__()

    def transform(self, dataset):
        pos_set = []
        neg_set = []
        for sample in dataset:
            if sample['Value'] == '0':
                neg_set.append(sample)
            else:
                pos_set.append(sample)

        pos_num =len(pos_set)
        neg_num = len(neg_set)
        print("The original dataset contains ", pos_num, " positive samples, ", neg_num, " negative samples.")

        if pos_num > neg_num:
            times = int((pos_num - neg_num) / neg_num)
            remainder = (pos_num - neg_num) % neg_num
            enlarged_neg_set = self._enlarge(neg_set, times, remainder)
            assert len(enlarged_neg_set) == pos_num
            enlarged_set = pos_set + enlarged_neg_set
        else:
            times = int((neg_num - pos_num) / pos_num)
            remainder = (neg_num - pos_num) % pos_num
            enlarged_pos_set = self._enlarge(pos_set, times, remainder)
            assert len(enlarged_pos_set) == neg_num
            enlarged_set = neg_set + enlarged_pos_set

        random.shuffle(enlarged_set)
        return enlarged_set

    def _enlarge(self, set, times, remainder):
        assert remainder < len(set)
        enlarged_set = []
        enlarged_set += set
        for i in range(times):
            enlarged_set += set
        random.shuffle(set)
        enlarged_set += set[:remainder]
        return enlarged_set
