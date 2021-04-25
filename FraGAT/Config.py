import os

class DefaultConfigs(object):
    def __init__(self):
        self.args = {
            'ExpName': None,
        }
    def set_args(self, argname, value):
        self.args[argname] = value

    def add_args(self, argname, value):
        if argname in self.args:
            self.set_args(argname, value)
        else:
            self.args.update({argname: value})


class Configs(DefaultConfigs):
    def __init__(self, BasicParamList):
        super(Configs, self).__init__()
        for param in BasicParamList.keys():
            self.add_args(param, BasicParamList.get(param))




class ConfigController(object):
    def __init__(self, BasicHyperparamList, AdjustableHyperparamList, SpecificHyperparamList = None):
        super(ConfigController,self).__init__()
        self.BasicHyperparameterList = BasicHyperparamList
        #name = self.BasicHyperparameterList['name']
        #main_metric = self.BasicHyperparameterList['MainMetric']
        self.HyperparameterList = AdjustableHyperparamList
        self.opt = Configs(self.BasicHyperparameterList)
        self.MainMetric = self.BasicHyperparameterList['MainMetric']

        if SpecificHyperparamList:
            self.SpecificHyperparamList = SpecificHyperparamList
            self.HyperparameterInit(self.SpecificHyperparamList)
        else:
            self.SpecificHyperparamList = None
            self.HyperparameterInit(self.HyperparameterList)

        self.exp_count = 0
        self.opt.add_args('TrialPath', self.opt.args['RootPath'] + self.opt.args['ExpName'] + '/')
        if not os.path.exists(self.opt.args['TrialPath']):
            os.mkdir(self.opt.args['TrialPath'])
        # set the Paths

        self.parampointer = 0
        self.paramvaluepointer = 1
        # two pointers indicates the param and its value that next experiment should use.
        self.result = []

    def HyperparameterInit(self, paramlist):
        for param in paramlist.keys():
            self.opt.add_args(param, paramlist.get(param)[0])
        # initially, the hyperparameters are set to be the first value of their candidate lists each.

    def StoreResults(self, score):
        self.result.append(score)

    def AdjustParams(self):
        if self.SpecificHyperparamList:
            keys = self.SpecificHyperparamList.keys()
            if self.exp_count < len(self.SpecificHyperparamList.get(list(keys)[0])):
                for param in self.SpecificHyperparamList.keys():
                    self.opt.set_args(param, self.SpecificHyperparamList.get(param)[self.exp_count])
                return False
            elif self.exp_count == len(self.SpecificHyperparamList.get(list(keys)[0])):
                self.HyperparameterInit(self.HyperparameterList)
                self.result = []
                return False

        ParamNames = list(self.HyperparameterList.keys())
        cur_param_name = ParamNames[self.parampointer]           # key, string
        cur_param = self.HyperparameterList[cur_param_name]      # list of values
        if self.paramvaluepointer < len(cur_param):
            # set the config
            cur_value = cur_param[self.paramvaluepointer]        # value
            self.opt.set_args(cur_param_name, cur_value)
            self.paramvaluepointer += 1
        else:
            # choose the best param value based on the results.
            assert len(self.result) == len(cur_param)

            if self.opt.args['ClassNum'] == 1:
                best_metric = min(self.result)
            else:
                best_metric = max(self.result)

            loc = self.result.index(best_metric)
            self.result = []
            self.result.append(best_metric)                      # best_metric is obtained by configs: {paraml:[loc], paraml+1:[0]}
                                                                 # so that we don't need to test the choice of paraml+1:[0]
                                                                 # just use the result tested when adjusting paraml.
            cur_param_best_value = cur_param[loc]
            self.opt.set_args(cur_param_name, cur_param_best_value)
            self.parampointer += 1
            self.paramvaluepointer = 1                           # test from paraml+1:[1]

            if self.parampointer < len(ParamNames):
                # set the config
                cur_param_name = ParamNames[self.parampointer]
                cur_param = self.HyperparameterList[cur_param_name]
                cur_value = cur_param[self.paramvaluepointer]
                self.opt.set_args(cur_param_name, cur_value)
                self.paramvaluepointer += 1
                return False
            else:
                return True


    def GetOpts(self):
        if not os.path.exists(self.opt.args['TrialPath'] + 'exp' + str(self.exp_count)):
            os.mkdir(self.opt.args['TrialPath'] + 'exp' + str(self.exp_count))

        self.opt.add_args('ExpDir', self.opt.args['TrialPath'] + 'exp' + str(self.exp_count) + '/')

        self.exp_count += 1
        return self.opt

    def LoadState(self, exp_count, parampointer, paramvaluepointer, result):
        self.exp_count = exp_count
        self.parampointer = parampointer
        self.paramvaluepointer = paramvaluepointer
        self.result = result
        print("Config Controller has been loaded. Experiments continue.")






