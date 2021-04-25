import json
import torch as t
import os
import re
from FraGAT.Config import *

class Saver(object):
    def __init__(self, opt):
        super(Saver, self).__init__()
        self.args = opt.args
        self.save_dir = self.args['SaveDir']
        if self.save_dir[-1] != '/':
            self.save_dir = self.save_dir + '/'
        self.ckpt_count = 0
        self.SaveConfig(opt)
        self.EarlyStopController = EarlyStopController(opt)

    def SaveConfig(self, opt):
        config_name = self.save_dir + 'config.json'
        with open(config_name, 'w') as f:
            json.dump(opt.args, f)

    def SaveModel(self, model, optimizer, epoch, scores, testscores):
        state = {'model': model, 'optimizer': optimizer, 'epoch': epoch}
        ckpt_name = self.save_dir + 'model/' + 'model_optimizer_epoch' + str(self.ckpt_count)
        t.save(state, ckpt_name)

        result_file_name = self.save_dir + 'result' + str(self.ckpt_count) + '.json'
        with open(result_file_name, 'w') as f:
            json.dump(scores, f)
        print("Model saved.")

        ShouldStop = self.EarlyStopController.ShouldStop(scores, self.ckpt_count, testscores)
        if ShouldStop:
            BestValue, BestModelCkpt, TestValue = self.EarlyStopController.BestModel()
            print("Early stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            print("The Test Value is: ", TestValue)
            # delete other models
            self.DeleteUselessCkpt(BestModelCkpt)
            return True, BestModelCkpt, BestValue
        else:
            self.ckpt_count += 1
            BestValue, BestModelCkpt, TestValue = self.EarlyStopController.BestModel()
            return False, BestModelCkpt, BestValue

    def DeleteUselessCkpt(self, BestModelCkpt):
        model_path = self.save_dir + 'model/'
        file_names = os.listdir(model_path)
        for file in file_names:
            ckpt_idx = re.split('model_optimizer_epoch', file)[-1]
            if int(ckpt_idx) != BestModelCkpt:
                exact_file_path = model_path + file
                os.remove(exact_file_path)

    def LoadModel(self, ckpt=None):
        dir_files = os.listdir(self.save_dir + 'model/')  # list of the checkpoint files
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.save_dir+'model/', x)))
            last_model_ckpt = dir_files[-1]   # find the latest checkpoint file.

            checkpoint = t.load(os.path.join(self.save_dir+'model/', last_model_ckpt))  # load the last checkpoint
            # the checkpoint include the state information of model, optimizer and epoch.
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            epoch = checkpoint['epoch']

            self.ckpt_count = int(re.split('epoch',last_model_ckpt)[-1]) + 1  # update the ckpt_count, get rid of overwriting the existed checkpoint files.
            return model, optimizer, epoch
        else:
            return None, None, None


class EarlyStopController(object):
    def __init__(self, opt):
        super(EarlyStopController, self).__init__()
        self.opt = opt
        self.MetricName = opt.args['MainMetric']
        if self.opt.args['ClassNum'] == 1:
            self.MaxResult = 9e8
        else:
            self.MaxResult = 0
        self.MaxResultModelIdx = None
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.LowerThanMaxLimit = opt.args['LowerThanMaxLimit']
        self.DecreasingLimit = opt.args['DecreasingLimit']
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx, testscore):
        MainScore = score[self.MetricName]
        MainTestScore = testscore[self.MetricName]
        self.TestResult.append(MainTestScore)
        if self.opt.args['ClassNum'] != 1:
            if MainScore > self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all set to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore < self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore
        else:
            if MainScore < self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all set to 0.
            else:
            # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore > self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        #print(self.MaxResultModelIdx)
        #print(self.TestResult)
        return self.MaxResult, self.MaxResultModelIdx, self.TestResult[self.MaxResultModelIdx]








