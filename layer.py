import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from torch_geometric.nn import RGATConv
import numpy as np
from utils import args


class StructureEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=6, n_heads=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.conv1 = RGATConv(768, 128, num_relations, heads=n_heads,concat=False, edge_dim=8)
        self.lin1 = torch.nn.Linear(128, 256)
        self.conv2 = RGATConv(256, 384, num_relations, heads=n_heads, concat=False, edge_dim=8)
        self.lin2 = torch.nn.Linear(384, 256)
        self.conv3 = RGATConv(256, 128, num_relations, heads=n_heads, concat=False, edge_dim=8)
        self.lin3 = torch.nn.Linear(128, 256)
        self.conv4 = RGATConv(256, 128, num_relations, heads=n_heads, concat=False, edge_dim=8)
        self.lin4 = torch.nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        x = data.x  # [node_num, dim]
        edge_index = data.edge_index.to(torch.long)  # [2, edge_num]
        edge_type = data.edge_type  # [edge_num, ]
        edge_attr = data.edge_attr
        x = self.conv1(x, edge_index, edge_type, edge_attr)
        x = self.lin1(x)
        x = self.conv2(x, edge_index, edge_type, edge_attr)
        x = self.lin2(x)
        x = self.conv3(x, edge_index, edge_type, edge_attr)
        x = self.lin3(x)
        x = self.conv4(x, edge_index, edge_type, edge_attr)
        x = self.lin4(x)
        x = self.dropout(x)
        batch_hole_idx = [idx + 1 for idx, i in enumerate(data.batch) if idx < len(data.batch) - 1 and i != data.batch[idx + 1]]
        batch_hole_idx.insert(0, 0)
        batch_hole_embedding = torch.index_select(x, 0, torch.tensor(batch_hole_idx))
        return F.softmax(batch_hole_embedding, dim=1)


class TextEncoder(nn.Module):
    def __init__(self, out_dim):
        super(TextEncoder, self).__init__()
        self.bert = RobertaForMaskedLM.from_pretrained("./codebert-base-mlm")
        # self.lm_head = nn.ModuleList()
        # self.lm_head.append(nn.Linear(768, 768, bias=True))
        # self.lm_head.append(nn.LayerNorm(768, eps=1e-5, elementwise_affine=True))
        # self.lm_head.append(nn.Linear(768, 50265, bias=True))
        for param in self.bert.parameters():  # nn.Module有成员函数parameters()
            param.requires_grad = False
        self.bert.lm_head.dense = nn.Linear(768, 768, bias=True)
        self.bert.lm_head.layer_norm = nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)
        self.bert.lm_head.decoder = nn.Linear(768, 50265, bias=True)
        # self.bert = RobertaForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
        self.tokenizer = RobertaTokenizer.from_pretrained("./codebert-base-mlm")
        # self.tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
        self.linear = nn.Linear(in_features=50265, out_features=out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    # inputs list [batch, src_len]  have not been padding
    def forward(self, inputs):
        inputs_new = []
        for input in inputs:
            inputs_new.append(" ".join(args.token_vocab.get(str(token.item())) for token in input))
        # inputs_new = ["".join(input).replace("\"", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
        #           .replace(".", "").replace(",", "").replace(";", "") for input in inputs_new]  # .replace(" ", "")
        new_inputs = self.tokenizer(inputs_new, return_tensors="pt", padding=True, truncation=True)  # batch_size list 'input_ids' 'attention_mask'
        logits = self.bert(**new_inputs).logits  # [batch_size, token_num, d_model(50265)]
        mask_token_indexs = [torch.tensor((i == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0], dtype=torch.long)
                             for i in new_inputs.input_ids]  # [batch_size, mask_num] different in each sentence
        # no_mask_idxs = [idx for idx, x in mask_token_indexs if x == torch.tensor([])]
        # if no_mask_idxs:
        #     # 若是不存在<mask> 则进行切割，
        #     # mask_token_indexs[idx] = torch.tensor([args.vocab_size-1])
        #     for idx in no_mask_idxs:
        #         no_mask_input = inputs[idx]
        #         no_mask_new_input = self.tokenizer(no_mask_input, return_tensors="pt", padding=True, truncation=True)

        mask_embeddings = logits[0, mask_token_indexs]  # [b, d_model]
        predicted_token_ids = [logits[0, i].argmax(axis=-1) for i in mask_token_indexs]
        predicted_tokens = [self.tokenizer.decode(i) for i in predicted_token_ids]
        mask_embeddings = self.linear(mask_embeddings)
        mask_embeddings = self.relu(mask_embeddings)
        mask_embeddings = self.dropout(mask_embeddings)
        return F.softmax(mask_embeddings, dim=1)  # [batch_size, output_dim]


class CodeEncoder(nn.Module):
    def __init__(self, out_dim, in_channels, hidden_channels, out_channels, num_relations, final_dim):
        super(CodeEncoder, self).__init__()
        self.structure_encoder = StructureEncoder(in_channels, hidden_channels, out_channels, num_relations)
        self.text_encoder = TextEncoder(out_dim)
        self.lin1 = nn.Linear(out_channels, final_dim)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(out_dim, final_dim)
        self.relu2 = nn.ReLU()
        self.lin = nn.Linear(out_channels + out_dim, args.vocab_size)

    def forward(self, data):
        # struct_hole_embedding = self.structure_encoder(data)  # [batch_size, 256]   hhh
        # struct_hole_embedding = self.lin1(struct_hole_embedding)
        # struct_hole_embedding = self.relu1(struct_hole_embedding)
        inputs = data['code']
        text_hole_embedding = self.text_encoder(inputs)  # [batch_size, 256]
        text_hole_embedding = self.lin2(text_hole_embedding)
        text_hole_embedding = self.relu2(text_hole_embedding)
        # hole_embedding = torch.concat((text_hole_embedding, struct_hole_embedding), dim=1)   hhh
        # hole_embedding = self.lin(hole_embedding)
        # hole_embedding = text_hole_embedding + struct_hole_embedding
        return F.softmax(text_hole_embedding, dim=1)


class EarlyStopping(nn.Module):
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, val_result, test_loss, test_result, model, path):
        # print("val_loss={:.5f}  test_loss={:.5f} val_acc_1={:.3f} val_acc_3={:.3f} val_acc_5={:.3f} val_acc_10={:.3f} "
        #       "test_acc_1={:.3f} test_acc_3={:.3f} test_acc_5={:.3f} test_acc_10={:.3f}".format(val_loss, test_loss,
        #     val_result[0], val_result[1], val_result[2], val_result[3], test_result[0], test_result[1], test_result[2],
        #                                                                                         test_result[3]))
        print("val_loss={:.5f}, acc1={:.3f}, acc3={:.3f}, acc5={:.3f}, acc10={:.3f}, map1={:.3f}, map3={:.3f}, "
              "map5={:.3f}, "
              " map10={:.3f}, mrr={:.3f}, ndcg_1={:.3f}".format(val_loss, val_result[0], val_result[1], val_result[2],
                                                              val_result[3], val_result[4], val_result[5], val_result[6],
                                                              val_result[7], val_result[8], val_result[9]))
        print("test_loss={:.5f}, acc1={:.3f}, acc3={:.3f}, acc5={:.3f}, acc10={:.3f}, map1={:.3f}, map3={:.3f}, "
              "map5={:.3f},  map10={:.3f}, mrr={:.3f}, ndcg_1={:.3f}".format(test_loss, test_result[0], test_result[1],
                                                                           test_result[2], test_result[3], test_result[4],
                                                                           test_result[5], test_result[6], test_result[7],
                                                                           test_result[8], test_result[9]))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print("\n\r")
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss