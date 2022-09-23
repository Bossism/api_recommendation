import json

from matplotlib import pyplot as plt
from torch_geometric.data import download_url, extract_zip, DataLoader
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# edge_path = 'data/packageName_edge.csv'
# node_path = 'data/packageName_node.csv'

# print(pd.read_csv(node_path).head())
# print(pd.read_csv(edge_path).head())


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    # 需要取出所有规定类型的token值，作为textencoder的输入
    # assign_nodes = [df.context[i].split("=")[0] for i, x in enumerate(df.label.values) if x == 'Assign']
    # for node in enumerate(df.label.values):

    mapping = {index: i for i, index in enumerate(df.index.unique())}
    hole_idx = [df.index[i] for i, x in enumerate(df.label.values) if x == "Hole"]   # hole_idx指的是node_csv文件中的hole对应的node_Id 有mapping之后好求在矩阵中的第几行
    x = None
    if encoders is not None:
        # for col, encoder in encoders.items():
        #     a = encoder(df[col])
        xs = [encoder(df[col]) for col, encoder in encoders.items()]  # list [30, 384]
        x = torch.cat(xs, dim=-1)  # [30, 768]

    return x, mapping, hole_idx


class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class EdgeClassEncoder(object):
    def __init__(self):
        pass

    def __call__(self, df):
        edge_classes_mapping = {'Data': '0', 'Output': '1', 'Ctrl-true': '2', 'Ctrl-false': '3', 'Call': '4',
                        'Param-in': '5', 'Param-out': '6', 'Member-of': '7'}
        # edge_classes_mapping = {edge_class: i for i, edge_class in edge_classes}
        edge_attr = torch.zeros(len(df), len(edge_classes_mapping))
        for i, col in enumerate(df.values):
            edge_attr[i, int(edge_classes_mapping[col])] = 1
        return edge_attr

# node_x, node_mapping, hole_index = load_node_csv(
#     node_path, index_col='nodeId', encoders={
#         'label': SequenceEncoder(),
#         'context': SequenceEncoder()
#     })


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, hole_index,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    src_hole = [i for i, x in enumerate(df.src.values) if x == hole_index]
    des_hole = [i for i, x in enumerate(df.des.values) if x == hole_index]   # src_hole des_hole 指的是符合条件的边是rdge_csv文件中的第几条边
    # (hole connect node)
    # (node connect hole)
    src = [index for index in df[src_index_col]]
    dst = [index for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)  # [43, 384]

    return edge_index, edge_attr, torch.tensor(src_hole), torch.tensor(des_hole)


def load_tokens_vocab():
    with open('tokens', 'r')as fp:   # encoding='utf8'
        json_data = json.load(fp)
        return json_data
        # print('这是文件中的json数据：',json_data)


class argparse():
    pass


global args
args = argparse()
args.epochs, args.learning_rate, args.patience = [100, 0.001, 4]
args.hidden_size, args.input_size, args.out_size = [128, 768, 256]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
args.vocab_size = 602
args.out_dim = 256
args.batch_size = 10
args.data = 2000
args.token_vocab = load_tokens_vocab()


def map_k(output, target, top_k):
    if len(target) == 0:
        return 0.0
    res = []
    _, pred = output.topk(top_k, 1, True, True)
    for i, t in enumerate(pred):
        score = 0.0
        num_hits = 0.0
        for j, p in enumerate(t):
            if p == target[i]:
                num_hits += 1.0
                score += num_hits / (j + 1.0)
                res.append(score / max(1.0, 1))  # len(gt[j])
    if len(res) == 0:
        return 0.0
    return np.mean(res)


def NDCG(output, target, top_k, use_graded_scores=False):
    score = 0.0
    _, pred = output.topk(top_k, 0, True, True)
    for rank, item in enumerate(pred):
        if item == target:
            if use_graded_scores:
                grade = 1.0 / (target.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    for rank in range(1):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def accuracy_k(output, target, top_k):
    # maxk = max(topk)
    maxk = top_k
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # res = []
    # for k in topk:
    correct_k = correct[:top_k].contiguous().view(-1).float().sum(0, keepdim=True) / batch_size
    # res.append(correct_k / batch_size)
    return correct_k


def metrics(gt, pred, metrics_map):
    out = np.zeros((len(metrics_map),), np.float32)

    if 'acc_1' in metrics_map:
        acc_1 = accuracy_k(pred, gt, 1)
        out[metrics_map.index('acc_1')] = acc_1

    if 'acc_3' in metrics_map:
        acc_3 = accuracy_k(pred, gt, 3)
        out[metrics_map.index('acc_3')] = acc_3

    if 'acc_5' in metrics_map:
        acc_5 = accuracy_k(pred, gt, 5)
        out[metrics_map.index('acc_5')] = acc_5

    if 'acc_10' in metrics_map:
        acc_10 = accuracy_k(pred, gt, 10)
        out[metrics_map.index('acc_10')] = acc_10

    if 'MAP_1' in metrics_map:
        map_1 = map_k(pred, gt, 1)
        out[metrics_map.index('MAP_1')] = map_1

    if 'MAP_3' in metrics_map:
        map_3 = map_k(pred, gt, 3)
        out[metrics_map.index('MAP_3')] = map_3

    if 'MAP_5' in metrics_map:
        map_5 = map_k(pred, gt, 5)
        out[metrics_map.index('MAP_5')] = map_5

    if 'MAP_10' in metrics_map:
        map_10 = map_k(pred, gt, 10)
        out[metrics_map.index('MAP_10')] = map_10

    if 'MRR' in metrics_map:
        res = []
        _, pred = pred.topk(10, 1, True, True)
        for i, item in enumerate(pred):
            score = 0.0
            for j, rank in enumerate(item):
                if rank == gt[i]:
                    score = 1.0 / (j + 1.0)
                    res.append(score)
                    break
        if len(res) == 0:
            out[metrics_map.index('MRR')] = 0.0
        else:
            out[metrics_map.index('MRR')] = np.mean(res)

    # if 'MRR@10' in metrics_map:
    #     score = 0.0
    #     for rank, item in enumerate(pred[:10]):
    #         if item in gt:
    #             score = 1.0 / (rank + 1.0)
    #             break
    #     out[metrics_map.index('MRR@10')] = score

    if 'NDCG_1' in metrics_map:
        ndcg = []
        for i, pred in enumerate(pred):
            tmp_ndcg = NDCG(pred, gt[i], 1)
            ndcg.append(tmp_ndcg)
        if len(ndcg) == 0:
            out[metrics_map.index('NDCG_1')] = 0.0
        else:
            out[metrics_map.index('NDCG_1')] = np.mean(ndcg)

    return out


def plot_loss(train_loss, train_epochs_loss, valid_epochs_loss):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('./text.jpg')
    # plt.show()