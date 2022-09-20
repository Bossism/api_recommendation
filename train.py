import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import args, metrics, plot_loss
import random
from torch_geometric.loader import DataLoader
from dataset import MyDataset
from layer import EarlyStopping, CodeEncoder

seed = 999
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


dataset = MyDataset(root="code/")
# [print(data.edge_attr.size()) for data in dataset]
# dataset.process()
dataset = dataset.shuffle()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
follow_batch_list = ['y', 'x', 'edge_index', 'edge_attr', 'code', 'node_type', 'edge_type']
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, follow_batch=follow_batch_list)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

model = CodeEncoder(out_dim=args.out_dim, in_channels=args.input_size, hidden_channels=args.hidden_size,
                    out_channels=args.out_size, num_relations=3, final_dim=args.vocab_size).to(args.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

train_loss = []
valid_loss = []
test_loss = []
train_epochs_loss = []
valid_epochs_loss = []
test_epochs_loss = []

# METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
METRICS_MAP = ['acc_1', 'acc_3', 'acc_5', 'acc_10', 'MAP_1', 'MAP_3', 'MAP_5', 'MAP_10', 'MRR', 'NDCG_1']
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

for epoch in range(args.epochs):
    model.train()
    train_epoch_loss = []
    for idx, data in enumerate(train_loader):
        # text = data['code']#.to(args.device)
        graph = data.to(args.device)
        y = data.y.to(args.device)
        y = np.nonzero(y == 1)[:, 1]
        # data_x = data_x.to(torch.float32).to(args.device)
        # data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(graph)
        # _, pred = outputs.max(dim=1)
        # correct = int(pred.eq(y).sum().item())
        # acc = correct / args.batch_size
        optimizer.zero_grad()
        loss = criterion(outputs.unsqueeze(dim=2).unsqueeze(dim=3), y.unsqueeze(1).unsqueeze(2))
        result = metrics(gt=y, pred=outputs, metrics_map=METRICS_MAP)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx % (len(train_loader) // 2) == 0:
        # print("epoch={}/{},{}/{}of train, loss={:.5f}, acc_1={:.3f}, acc_3={:.3f}, acc_5={:.3f}, acc_10={:.3f}".format(
        #     epoch, args.epochs, idx, len(train_loader), loss.item(), result[0], result[1], result[2], result[3]))
            print("epoch={}/{},{}/{}of train, train_loss={:.5f}, acc_1={:.3f}, acc_3={:.3f}, acc_5={:.3f}, acc_10={:.3f}, MAP_1={:.3f}, "
                "MAP_3={:.3f}, MAP_5={:.3f}, MAP_10={:.3f}, MRR={:.3f}, NDCG_1={:.3f}".format(epoch, args.epochs, idx, len(train_loader), loss.item(),
                    result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9]))
    train_epochs_loss.append(np.average(train_epoch_loss))

    # =====================valid============================
    model.eval()
    valid_epoch_loss = []
    # for idx, (data_x, data_y) in enumerate(val_loader, 0):
    for idx, data in enumerate(val_loader):
        # text = data['code']#.to(args.device)
        graph = data.to(args.device)
        y = data.y.to(args.device)
        y = np.nonzero(y == 1)[:, 1]
        # data_x = data_x.to(torch.float32).to(args.device)
        # data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(graph)
        loss = criterion(outputs, y)
        valid_result = metrics(gt=y, pred=outputs, metrics_map=METRICS_MAP)
        # _, pred = outputs.max(dim=1)
        # correct = int(pred.eq(y).sum().item())
        # valid_acc = correct / args.batch_size
        #     return acc, loss.item()
        # result = metrics(gt=y, pred=outputs, metrics_map=METRICS_MAP)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    # ==================test======================
    model.eval()
    test_epoch_loss = []
    for idx, data in enumerate(test_loader):
        # text = data['code']  # .to(args.device)
        graph = data.to(args.device)
        y = data.y.to(args.device)
        y = np.nonzero(y == 1)[:, 1]
        # data_x = data_x.to(torch.float32).to(args.device)
        # data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(graph)
        loss = criterion(outputs, y)
        test_result = metrics(gt=y, pred=outputs, metrics_map=METRICS_MAP)
        # _, pred = outputs.max(dim=1)
        # correct = int(pred.eq(y).sum().item())
        # test_acc = correct / args.batch_size
        #     return acc, loss.item()
        # result = metrics(gt=y, pred=outputs, metrics_map=METRICS_MAP)
        test_epoch_loss.append(loss.item())
        test_loss.append(loss.item())
    test_epochs_loss.append(np.average(test_epoch_loss))

    plot_loss(train_loss, train_epochs_loss, valid_epochs_loss)

    # ==================early stopping======================
    early_stopping(valid_epochs_loss[-1], valid_result, test_epochs_loss[-1], test_result, model=model, path=r'model/')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    # ====================adjust lr========================
    lr_adjust = {
        2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        10: 5e-7, 15: 1e-7, 20: 5e-8
    }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# def ceshi():
#     model.eval()
#     for idx, data in enumerate(test_loader):
#         text = data['code']  # .to(args.device)
#         graph = data.to(args.device)
#         y = data.y.to(args.device)
#         y = np.nonzero(y == 1)[:, 1]
#         # data_x = data_x.to(torch.float32).to(args.device)
#         # data_y = data_y.to(torch.float32).to(args.device)
#         outputs = model(text, graph)
#         loss = criterion(outputs, y)
#         _, pred = outputs.max(dim=1)
#         correct = int(pred.eq(y).sum().item())
#         acc = correct / args.batch_size
