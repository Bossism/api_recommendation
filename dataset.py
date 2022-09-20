import glob
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data, HeteroData, DataLoader
from utils import load_node_csv, SequenceEncoder, load_edge_csv, EdgeClassEncoder


class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.root + "\\raw"

    @property
    def processed_file_names(self):
        processed_file_list = []
        for root, dirs, graphs_files in os.walk("code\\processed\\"):
            for files in dirs:
                for home, dir, file in os.walk("code\\processed\\" + files):
                    processed_file_list.extend([files + "\\" + file_ for file_ in file])

        pre_list = [name for name in os.listdir("code\\processed")
                 if os.path.isfile(os.path.join("code\\processed\\", name))]
        processed_file_list.extend(pre_list)
        return processed_file_list

    def download(self):
        pass

    def process(self):
        idx = 0
        graphs_idx = -1
        # raw_data_path = "csvs"
        for home, dirs, files in os.walk(self.raw_file_names):
            for graphs_dir in dirs:
                # 创建一个pt存储时的文件夹
                graphs_idx += 1
                graphs_path = os.path.join(self.processed_dir, f'graphs_{graphs_idx}')
                os.mkdir(graphs_path)
                for graphs_home, graphs_dir, graphs_files in os.walk(home + "\\" + graphs_dir):
                    for dir in graphs_dir:
                        graph_path = os.path.join(graphs_home, dir)

                        node_file_path = graph_path + '\\*_node*'
                        node_file_path = glob.glob(node_file_path)[0]
                        node_x, node_mapping, hole_index = load_node_csv(
                            node_file_path, index_col='nodeId', encoders={
                                'label': SequenceEncoder(),
                                'context': SequenceEncoder()
                            })
                        # 需要取出所有的token

                        edge_file_path = graph_path + '\\*_edge*'
                        edge_file_path = glob.glob(edge_file_path)[0]
                        # 建边
                        edge_index, edge_label, src_hole, des_hole = load_edge_csv(
                            edge_file_path,
                            src_index_col='src',
                            src_mapping=node_mapping,
                            dst_index_col='des',
                            dst_mapping=node_mapping,
                            hole_index=hole_index,
                            encoders={'label': EdgeClassEncoder()},
                        )

                        # ground_truth_file_path = graph_path + "\\groundtruth*"
                        # ground_truth_file_path = glob.glob(ground_truth_file_path)[0]
                        # f = open(ground_truth_file_path)
                        # for line in f:
                        #     ground_truth = line.strip()

                        text_file_path = graph_path + "\\*txt"
                        text_file_path = glob.glob(text_file_path)
                        for s in text_file_path:
                            if s.endswith("groundtruth.txt"):
                                gt_f = open(s, encoding='utf-8')
                                gt = gt_f.readline()
                            else:
                                text_file_path = s
                        f = open(text_file_path, encoding='utf-8')
                        text = f.readlines()

                        data = HeteroData()
                        data['hole'].x = node_x[node_mapping[hole_index[0]]].unsqueeze(0)  # [1, 768]
                        data['node'].x = torch.cat((node_x[:node_mapping[hole_index[0]]], node_x[node_mapping[hole_index[0]] + 1:]), 0)  # [29, 768]
                        # TODO y groundTruth中的答案在vocab中的idx
                        gt_emb = torch.zeros(1, 602)
                        gt_emb[:, int(gt)] = 1
                        data.y = gt_emb
                        data['code'] = text
                        # data['hole'].train_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
                        # data['hole'].val_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
                        # data['hole'].test_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

                        # edge_index_sub_one = edge_index - 1
                        edge_index = torch.where(edge_index == hole_index[0], 0, edge_index)
                        # edge_index = torch.where(edge_index > hole_index[0], edge_index_sub_one, edge_index)

                        if min(des_hole.shape) == 0:
                            data['node', 'connect', 'hole'].edge_index = torch.zeros(2, 0)
                            data['node', 'connect', 'hole'].edge_attr = torch.zeros(0, 8)
                        else:
                            data['node', 'connect', 'hole'].edge_index = torch.index_select(edge_index, 1, des_hole)
                            data['node', 'connect', 'hole'].edge_attr = torch.index_select(edge_label, 0, des_hole)

                        if min(src_hole.shape) == 0:
                            data['hole', 'connect', 'node'].edge_index = torch.zeros(2, 0)
                            data['hole', 'connect', 'node'].edge_attr = torch.zeros(0, 8)
                        else:
                            data['hole', 'connect', 'node'].edge_index = torch.index_select(edge_index, 1, src_hole)
                            data['hole', 'connect', 'node'].edge_attr = torch.index_select(edge_label, 0, src_hole)

                        node_node_index = np.delete(edge_index, des_hole.tolist() + src_hole.tolist(), axis=1)
                        node_node_label = np.delete(edge_label, src_hole.tolist() + des_hole.tolist(), axis=0)
                        data['node', 'connect', 'node'].edge_index = node_node_index
                        data['node', 'connect', 'node'].edge_attr = node_node_label
                        # edge_label [43, 384] node_type [30]  edge_type [43]

                        homogeneous_data = data.to_homogeneous()  # x [30, 768]  y [1, 1000], edge_index [2, 43] code {38}

                        edge_index_sub_one = homogeneous_data.edge_index - 1
                        homogeneous_data.edge_index = torch.where(homogeneous_data.edge_index > hole_index[0], edge_index_sub_one, homogeneous_data.edge_index)
                        # edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
                        # for i, x in enumerate(edge_index[0]):   #  不知道为啥hole的nodeIdx不变，不知道自己改掉对不对
                        #     if edge_type[i] == 0:
                        #         edge_index[0][i] -= 1
                        # for i, x in enumerate(edge_index[1]):
                        #     if edge_type[i] == 1:
                        #         edge_index[1][i] -= 1
                        # for i, x in enumerate(edge_index[0]):
                        #     if edge_type[i] == 2:
                        #         edge_index[0][i] -= 1
                        #         edge_index[1][i] -= 1
                        #
                        # print(data)
                        # print("+++++++++++++++++++++++++++++++++++++++++++++++++")

                        if self.pre_filter is not None and not self.pre_filter(homogeneous_data):
                            continue

                        if self.pre_transform is not None:
                            homogeneous_data = self.pre_transform(homogeneous_data)

                        torch.save(homogeneous_data, os.path.join(graphs_path, f'graph_{idx}.pt'))
                        # model = torch.load("code/processed/graph_0.pt") self.processed_dir, f'graphs_{graphs_idx}.pt'
                        idx += 1

    def len(self):
        return len(self.processed_file_names) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir + f'\\graphs_{idx // 1000}\\', f'graph_{idx}.pt'))
        return data
