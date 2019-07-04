import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.bert_emb import get_emb

class WordEmbedding(nn.Module):
    def __init__(self, N_word, gpu, SQL_TOK, our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable #是否自己训练emb
        self.N_word = N_word #word_emb的大小
        self.our_model = our_model 
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK
        
        '''
        if trainable:
            print ("Using trainable embedding")
            #self.w2i, word_emb_val = word_emb
            i=3
            self.w2i={"<BEG>":1,"END":2}
            word_emb_val=[[0 for _ in range(N_word)]for i in range(2)]
            for index,emb_val in word_emb.items():
                self.w2i[index]=i
                word_emb_val.append(emb_val)
                i+=1
            word_emb_val=np.array(word_emb_val)
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print ("Using fixed embedding")
        '''
    def gen_x_batch(self, q, col):
        B = len(q)
        max_len = 0
        for i in q:
            max_len = max_len if max_len >= len(i) else len(i)
        max_len+=2
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        x_emb , _ = get_emb(q,max_len,gpu=self.gpu)
        #for i, (one_q, one_col) in enumerate(zip(questions, col)):
        for i in range(B):
            if self.trainable:
                q_val = [self.w2i.get(x,0) for x in one_q]
                #print('<BEG>' in self.w2i.keys())
                val_embs.append([1]+q_val+[2])  #<BEG> and <END>
            else:
                # print (i)
                # print ([x.encode('utf-8') for x in one_q])
                q_val = x_emb[i][1:-1]
                #q_val = [x_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_q]
                # print (q_val)
                # print ("#"*60)
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
            # exit(0)
            val_len[i] = len(q_val) + 2
        max_len = max(val_len)
    
        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)
        max_len = 0
        for i in str_list:
            max_len = max_len if max_len >= len(i) else len(i)
        val_embs = []
        
        col_emb, _ = get_emb(str_list,max_len+2,gpu=self.gpu)
        val_len = np.zeros(B, dtype=np.int64)
        #for i, one_str in enumerate(str_list):
        for i in range(B):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = col_emb[i][1:-1]
                #val = [col_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len=max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
