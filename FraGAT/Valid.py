import torch as t
from torch.utils import data
from FraGAT.Metrics import *




def Validation(validloader, model, metrics, opt):
    model.eval()
    All_answer = []
    All_label = []
    for i in range(opt.args['TaskNum']):
        All_answer.append([])
        All_label.append([])
    # [tasknum, ]

    for ii, data in enumerate(validloader):
        # one molecule input, but batch is not 1. Different Frags of one molecule consist of a batch.
        [Input, Label] = data
        #[[AdjMat, FeatureMat], Label] = data
        #AdjMat = AdjMat.cuda()
        #FeatureMat = FeatureMat.cuda()

        Label = Label.cuda()  #     [wrongbatch, batch(mol), task, 1]
        Label = Label.squeeze(-1)   #[wrongbatch, batch(mol), task]
        Label = Label.squeeze(0)    #[batch(mol), task]
        #print(Label.size())
        Label = Label.t()           #[task,batch(mol)]
        # for Label, different labels in a batch are actually the same, for they are exactly one molecule.
        # so the batch dim of Label is not exactly useful.

        output = model(Input)       #[batch, output_size]
        #print("Pred value before average is: ", output)
        output = output.mean(dim=0, keepdims=True)    #[1, output_size]

        #print("Target value is: ", Label)
        #print("Pred value is: ", output)
        #print(output.size())
        #output = model(AdjMat, FeatureMat)
        #print(output)

        for i in range(opt.args['TaskNum']):
            cur_task_output = output[:, i * opt.args['ClassNum'] : (i+1) * opt.args['ClassNum']]    # [1, ClassNum]
            cur_task_label = Label[i][0]   # all of the batch are the same, so only picking [i][0] is enough.
            if cur_task_label == -1:
                continue
            else:
                All_label[i].append(cur_task_label.item())
                for ii, data in enumerate(cur_task_output.tolist()):
                    All_answer[i].append(data)
    scores = {}
    All_metrics = []
    #print(len(All_answer[0]))
    #print(len(All_label[0]))
    for i in range(opt.args['TaskNum']):
        All_metrics.append([])
        label = All_label[i]
        answer = All_answer[i]
        #print("Target value is: ", label)
        #print("Pred value is: ", answer)
        assert len(label) == len(answer)
        for metric in metrics:
            result = metric.compute(answer, label)
            All_metrics[i].append(result)
            if opt.args['TaskNum'] != 1:
                print("The value of metric", metric.name, "in task", i, 'is: ', result)

    #print(All_metrics)
    average = t.Tensor(All_metrics).mean(dim=0)
    for i in range(len(metrics)):
        scores.update({metrics[i].name: average[i].item()})
        print("The average value of metric", metrics[i].name, "is: ", average[i].item())
    model.train()
    return scores
'''
    scores = {}
    for metric in metrics:
        result = metric.compute(answer, label)
        scores.update({metric.name: result})
        print("The value of metric", metric.name, "is: ", result)
'''






