import torch.optim as optim
from torch.utils import data
from FraGAT.Model import *
from FraGAT.Dataset import *
from FraGAT.Valid import *
from FraGAT.Metrics import *
from FraGAT.Config import *
from FraGAT.Utils import *


def train_and_evaluate(opt):

    t.manual_seed(opt.args['TorchSeed'])
    saver = Saver(opt)
    net, optimizer, StartEpoch = saver.LoadModel()

    if not net:
        if opt.args['Feature'] == 'Graph':
            net = MolPredGraph(feature_size=39, GCN_layers=opt.args['GCNLayers'], DNN_layers=opt.args['DNNLayers'],
                               GCNLayertype=opt.args['GCNLayertype'])
        elif opt.args['Feature'] == 'FP':
            net = MolPredFP(feature_size=opt.args['feature_size'], DNN_layers=opt.args['DNNLayers'])
        elif opt.args['Feature'] == 'AttentiveFP':
            if opt.args['Frag'] == True:
                net = MolPredFragFPv8(
                    atom_feature_size=opt.args['atom_feature_size'],
                    bond_feature_size=opt.args['bond_feature_size'],
                    FP_size=opt.args['FP_size'],
                    atom_layers=opt.args['atom_layers'],
                    mol_layers=opt.args['mol_layers'],
                    DNN_layers=opt.args['DNNLayers'],
                    output_size=opt.args["output_size"],
                    drop_rate=opt.args['drop_rate'],
                    opt=opt
                )
            else:
                net = MolPredAttentiveFP(
                    atom_feature_size=opt.args['atom_feature_size'],
                    bond_feature_size=opt.args['bond_feature_size'],
                    FP_size=opt.args['FP_size'],
                    atom_layers=opt.args['atom_layers'],
                    mol_layers=opt.args['mol_layers'],
                    DNN_layers=opt.args['DNNLayers'],
                    output_size=opt.args["output_size"],
                    drop_rate=opt.args['drop_rate']
                )

    net = net.cuda()

    moldatasetcreator = MolDatasetCreator(opt)
    if len(opt.args['SplitRate']) == 1:
        (Trainset, Validset), weights = moldatasetcreator.CreateDatasets()
        Testset = Validset
    elif len(opt.args['SplitRate']) == 2:
        (Trainset, Validset, Testset), weights = moldatasetcreator.CreateDatasets()

    print(net)

    if opt.args['BatchSize'] == -1:
        batchsize = len(Trainset)
    else:
        batchsize = opt.args['BatchSize']

    trainloader = t.utils.data.DataLoader(Trainset, batch_size=batchsize, shuffle=True, num_workers=8, \
                                          drop_last=True, worker_init_fn=np.random.seed(8), pin_memory=True)
    validloader = t.utils.data.DataLoader(Validset, batch_size=1, shuffle=False, num_workers=0, \
                                          drop_last=True, worker_init_fn=np.random.seed(8), pin_memory=True)
    testloader = t.utils.data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, \
                                         drop_last=True, worker_init_fn=np.random.seed(8), pin_memory=True)
    if opt.args['ClassNum'] == 2:
        if opt.args['Weight']:
            criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction='mean').cuda() for weight in weights]
        else:
            criterion = [nn.CrossEntropyLoss().cuda() for i in range(opt.args['TaskNum'])]
    elif opt.args['ClassNum'] == 1:
        criterion = [nn.MSELoss().cuda() for i in range(opt.args['TaskNum'])]

    if not optimizer:
        optimizer = optim.Adam(net.parameters(), lr=10**-opt.args['lr'], weight_decay=10**-opt.args['WeightDecay'])

    net.zero_grad()
    optimizer.zero_grad()

    if not StartEpoch:
        StartEpoch = 0

    # init_result = Validation(validloader, net, AUC())
    # for name, params in net.named_parameters():
    #    print(name)
    #    print(params)
    print("Start Training...")
    stop_flag = False

    epoch = StartEpoch
    while epoch < opt.args['MaxEpoch']:
        print("Epoch: ", epoch)
        cum_loss = 0.0

        if stop_flag:
            break

        for ii, data in enumerate(trainloader):

            if stop_flag:
                break

            [Input, Label] = data
            Label = Label.cuda()
            Label = Label.squeeze(-1)    # [batch, task]
            Label = Label.t()            # [task, batch]


            output = net(Input)   # [batch, output_size]
            loss = 0.0
            if opt.args['ClassNum'] != 1:
                for i in range(opt.args['TaskNum']):
                    cur_task_output = output[:, i * opt.args['ClassNum'] : (i+1) * opt.args['ClassNum']]
                    cur_task_label = Label[i]   # [batch]
                    valid_index = (cur_task_label != -1)
                    valid_label = cur_task_label[valid_index]
                    if len(valid_label) == 0:
                        continue
                    else:
                        valid_output = cur_task_output[valid_index]
                        loss += criterion[i](valid_output, valid_label)
            else:
                for i in range(opt.args['TaskNum']):
                    cur_task_output = output[:, i * opt.args['ClassNum'] : (i+1) * opt.args['ClassNum']]
                    cur_task_label = Label[i].unsqueeze(-1)
                    loss += criterion[i](cur_task_output, cur_task_label)

            loss.backward()

            if (ii + 1) % opt.args['UpdateRate'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_loss += loss.detach()

            if (ii + 1) % opt.args['PrintRate'] == 0:
                print("Loss: ", cum_loss.item() / opt.args['PrintRate'])
                cum_loss = 0.0

            if (ii + 1) % opt.args['ValidRate'] == 0:
                if opt.args['ClassNum'] == 1:
                    result = Validation(validloader, net, [MAE(), RMSE()], opt)
                    stop_flag, best_ckpt, best_value = saver.SaveModel(net, optimizer, epoch, result)
                else:
                    result = Validation(validloader, net, [AUC(), ACC()])
                    stop_flag, best_ckpt, best_value = saver.SaveModel(net, optimizer, epoch, result)

        if opt.args['ClassNum'] == 1:
            result = Validation(validloader, net, [MAE(), RMSE()], opt)
            print("running on test set.")
            testresult = Validation(testloader, net, [MAE(), RMSE()], opt)

            stop_flag, best_ckpt, best_value = saver.SaveModel(net, optimizer, epoch, result, testresult)
            epoch += 1
        else:
            result = Validation(validloader, net, [AUC(), ACC()], opt)
            print("running on test set.")
            testresult = Validation(testloader, net, [AUC(), ACC()], opt)

            stop_flag, best_ckpt, best_value = saver.SaveModel(net, optimizer, epoch, result, testresult)
            epoch += 1


    return best_ckpt, best_value