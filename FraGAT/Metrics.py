import torch as t
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class ACC(object):
    def __init__(self):
        super(ACC, self).__init__()
        self.name = 'ACC'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        total = len(answer)

        answer = t.Tensor(answer)
        label = t.Tensor(label)

        pred = t.argmax(answer, dim=1)
        correct = sum(pred == label).float()
        acc = correct / total


        return acc.item()

class AUC(object):
    def __init__(self):
        super(AUC, self).__init__()
        self.name = 'AUC'

    def compute(self, answer, label):
        assert len(answer) == len(label)

        answer = t.Tensor(answer)
        answer = answer[:,1]
        answer = answer.tolist()

        #print(answer)
        #print(label)

        result = roc_auc_score(y_true = label, y_score= answer)
        return result

class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)
        #print("Size for MAE")
        #print("Answer size: ", answer.size())
        #print("Label size: ", label.size())
        MAE = F.l1_loss(answer, label, reduction = 'mean')
        return MAE.item()
'''
class MSE(object):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer)
        label = t.Tensor(label)
        MSE = F.mse_loss(answer, label, reduction = 'mean')
        return MSE.item()


class RMSE_hand(object):
    def __init__(self):
        super(RMSE_hand, self).__init__()
        self.name = 'RMSE_hand'

    def compute(self, answer, label):
        cum_error = 0.0
        assert len(answer) == len(label)
        for i in range(len(answer)):
            pred = answer[i][0]
            target = label[i]
            #print(pred[0])
            #print(target)
            error = abs(target - pred)
            squareerror = error ** 2
            cum_error += squareerror
        mean_error = cum_error / len(answer)
        R_mean_error = mean_error ** 0.5
        return R_mean_error

'''
class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)
        #print("Size for RMSE")
        #print("Answer size: ", answer.size())
        #print("Label size: ", label.size())
        RMSE = F.mse_loss(answer, label, reduction = 'mean').sqrt()
        #SE = F.mse_loss(answer, label, reduction='none')
        #print("SE: ", SE)
        #MSE = SE.mean()
        #print("MSE: ", MSE)
        #RMSE = MSE.sqrt()
        return RMSE.item()
