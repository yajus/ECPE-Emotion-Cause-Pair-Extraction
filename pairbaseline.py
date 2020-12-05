# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import argparse
import torch
import torch.nn
from torch import nn
import torchvision
import os,sys
from data_process import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
# FLAGS = tf.app.flags.FLAGS
# # >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# ## embedding parameters ##
# tf.app.flags.DEFINE_string('w2v_file', '../data/w2v_200.txt', 'embedding file')
# tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
# tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
# ## input struct ##
# tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
# ## model struct ##
# tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
# tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# # >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
# tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# # >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
# tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
# tf.app.flags.DEFINE_string('scope', 'P_cause', 'RNN scope')
# # not easy to tune , a good posture of using data to train model is very important
# tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
# tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
# tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
# tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
# tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--w2v_file', type=str, default='../data/w2v_200.txt',help='embedding file')
parser.add_argument('--embedding_dim', type=int, default=200, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, help='dimension of position embedding')
parser.add_argument('--max_sen_len', type=int, default=30, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=75, help='max number of tokens per documents')
##### model struct
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--log_file_name', type=str, default='', help='name of log file')
##### traing
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='RNN', help='scope')
parser.add_argument('--batch_size', type=int, default=32, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
# parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')

parser.add_argument('--keep_prob1', type=float, default=0.8, help='word embedding training dropout keep prob')
parser.add_argument('--keep_prob2', type=float, default=1.0, help='softmax layer dropout keep prob')
parser.add_argument('--l2_reg', type=float, default=0.00001, help='l2 regularization')
parser.add_argument('--cause', type=float, default=1.000, help='lambda1')
parser.add_argument('--pos', type=float, default=1.00, help='lambda2')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
opt = parser.parse_args()

if opt.usegpu and torch.cuda.is_available():
    use_gpu=True
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


class pairDataset(Dataset):
    def __init__(self, input_file,word_idx, batchsize= opt.batch_size, max_doc_len = 75, max_sen_len = 30,test=False,transforms=None):
        self.transforms=transforms #NLP中实际上没用到
        print('load data_file: {}'.format(input_file))
        self.pair_id_all, self.pair_id, self.y, self.x, self.sen_len, self.distance = [], [], [], [], [], []
        self.batch_size=batchsize
        self.test=test
        self.n_cut = 0
        self.doc_id = []
        inputFile = open(input_file, 'r')
        while True:
            line = inputFile.readline()
            if line == '': break
            line = line.strip().split()
            doc_id = int(line[0])
            d_len = int(line[1])
            pairs = eval(inputFile.readline().strip())
            self.pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])
            sen_len_tmp, x_tmp = np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
            pos_list, cause_list = [], []
            for i in range(d_len):
                line = inputFile.readline().strip().split(',')
                if int(line[1].strip()) > 0:
                    pos_list.append(i + 1)
                if int(line[2].strip()) > 0:
                    cause_list.append(i + 1)
                words = line[-1]
                sen_len_tmp[i] = min(len(words.split()), max_sen_len)
                for j, word in enumerate(words.split()):
                    if j >= max_sen_len:
                        self.n_cut += 1
                        break
                    x_tmp[i][j] = int(word_idx[word])
            for i in pos_list:
                for j in cause_list:
                    pair_id_cur = doc_id * 10000 + i * 100 + j
                    self.pair_id.append(pair_id_cur)
                    self.y.append([0, 1] if pair_id_cur in self.pair_id_all else [1, 0])
                    self.x.append([x_tmp[i - 1], x_tmp[j - 1]])
                    self.sen_len.append([sen_len_tmp[i - 1], sen_len_tmp[j - 1]])
                    self.distance.append(j - i + 100)  # ？
        self.y, self.x, self.sen_len, self.distance = map(np.array, [self.y, self.x, self.sen_len, self.distance])
        for var in ['self.y', 'self.x', 'self.sen_len', 'self.distance']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}, (y-negative, y-positive): {}'.format(self.n_cut, self.y.sum(axis=0)))
        print('load data done!\n')
        self.index = [i for i in range(len(self.y)) ]

        # return self.pair_id_all, self.pair_id, self.y, self.x, self.sen_len, self.distance

    def __getitem__(self, index):
        index = self.index[index]
        # feed_list = [self.x[index], self.sen_len[index], self.doc_len[index], opt.keep_prob1, opt.keep_prob2, self.y_position[index],
                       # self.y_cause[index]]
        feed_list = [self.x[index], self.sen_len[index], opt.keep_prob1, opt.keep_prob2, self.distance[index], self.y[index]]

        return feed_list


    def __len__(self):
        return len(self.x)


class attention(torch.nn.Module):
    def __init__(self,input_size,hidden_size):
        super(attention, self).__init__()
        self.bilstm = nn.LSTM(input_size = input_size,hidden_size=hidden_size,bidirectional=True,batch_first=True)
        self.sh2 = 2 * opt.n_hidden
        self.linearlayer1 = nn.Linear(self.sh2,self.sh2).cuda()
        self.linearlayer2 = nn.Linear(self.sh2,1,bias=False).cuda()
        # self.w1 = get_weight_varible([self.sh2, self.sh2])
        # self.b1 = get_weight_varible([self.sh2])
        # self.w2 = get_weight_varible([self.sh2, 1])
        # if use_gpu:
        #     self.w1= self.w1.cuda()
        #     self.b1 = self.b1.cuda()
        #     self.w2 = self.w2.cuda()


    def forward(self,inputs,sen_len):
        r_out, (h_n, h_c) = self.bilstm(inputs)
        inputs = r_out
        self.sen_len = sen_len
        max_len, n_hidden = (inputs.shape[1], inputs.shape[2])
        tmp = torch.reshape(inputs, [-1, n_hidden])
        u = torch.tanh(self.linearlayer1(tmp))
        alpha = torch.reshape(self.linearlayer2(u), [-1, 1, max_len])  # 200->1
        alpha = softmax_by_length(alpha, self.sen_len, use_gpu)
        s = torch.reshape(torch.matmul(alpha, inputs), [-1, n_hidden])
        s = torch.reshape(s, [-1,  2 * 2 * opt.n_hidden])
        return s

class pairmodel(torch.nn.Module):
    def __init__(self, word_embedding, pos_embedding, keep_prob1, keep_prob2,input_size=opt.n_hidden*2,hidden_size=opt.n_hidden):
        super(pairmodel, self).__init__()

        # self.causebilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        # self.posbilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.attention = attention(input_size=input_size, hidden_size=hidden_size)
        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding

        self.nnlayer = nn.Linear(4 * opt.n_hidden + opt.embedding_dim_pos, opt.n_class).cuda()
        # self.nnlayer_pos = nn.Linear(2 * opt.n_hidden, opt.n_class).cuda()


    def forward(self, x,sen_len,keep_prob1,keep_prob2,distance):
        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2
        self.sen_len = sen_len
        # self.doc_len = doc_len
        x = x.long()
        x = torch.index_select(self.word_embedding,dim=0,index = torch.reshape(x,[-1]))
        if use_gpu:
            x = x.cuda()
        inputs = torch.reshape(x,[-1,opt.max_sen_len,opt.embedding_dim])
        inputs = torch.nn.Dropout(1.0 - self.keep_prob1)(inputs)
        sen_len = torch.reshape(self.sen_len, [-1])
        s = self.attention(inputs,sen_len)
        dis = torch.index_select(self.pos_embedding, dim=0, index=torch.reshape(distance, [-1]))
        if use_gpu:
            dis = dis.cuda()
        s = torch.cat([s,dis],1)
        s1 = torch.nn.Dropout(1.0 - self.keep_prob2)(s)
        pred_pair  = F.softmax(self.nnlayer(s1))
        reg = torch.norm(self.nnlayer.weight) + torch.norm(self.nnlayer.bias)
        return pred_pair, reg


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        opt.batch_size, opt.learning_rate, opt.keep_prob1, opt.keep_prob2, opt.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(opt.training_iter, opt.scope))

def run():
    save_dir = 'pair_data/{}/'.format(opt.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if opt.log_file_name:
        sys.stdout = open(save_dir + opt.log_file_name, 'w')
    print_time()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(opt.embedding_dim,opt.embedding_dim_pos,'data_combine/clause_keywords.csv',opt.w2v_file)
    word_embedding = torch.FloatTensor(word_embedding)
    pos_embedding = torch.FloatTensor(pos_embedding)

    # print('build model...')

    # x = tf.placeholder(tf.int32, [None, 2, FLAGS.max_sen_len])
    # sen_len = tf.placeholder(tf.int32, [None, 2])
    # keep_prob1 = tf.placeholder(tf.float32)
    # keep_prob2 = tf.placeholder(tf.float32)
    # distance = tf.placeholder(tf.int32, [None])
    # y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
    # placeholders = [x, sen_len, keep_prob1, keep_prob2, distance, y]

    # pred_pair, reg = build_model(word_embedding, pos_embedding, x, sen_len, keep_prob1, keep_prob2, distance, y)
    # loss_op = - tf.reduce_mean(y * tf.log(pred_pair)) + reg * FLAGS.l2_reg
    # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    # true_y_op = tf.argmax(y, 1)
    # pred_y_op = tf.argmax(pred_pair, 1)
    # acc_op = tf.reduce_mean(tf.cast(tf.equal(true_y_op, pred_y_op), tf.float32))
    # print('build model done!\n')

    # Training Code Block
    print_training_info()
# with tf.Session(config=tf_config) as sess:
    keep_rate_list, acc_subtask_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], [], []
    o_p_pair_list, o_r_pair_list, o_f1_pair_list = [], [], []

    for fold in range(1, 11):
        print('build model..')
        model = pairmodel(word_embedding, pos_embedding,opt.keep_prob1, opt.keep_prob2, opt.n_hidden * 2, opt.n_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

        print('build model end..')
        if use_gpu:
            model = model.cuda()
        # train for one fold
        print('############# fold {} begin ###############'.format(fold))
        # Data Code Block
        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)

        train = save_dir + train_file_name
        test = save_dir + test_file_name
        edict = {"train":train,"test":test}


        NLP_Dataset = {x:pairDataset(edict[x],word_id_mapping,batchsize=opt.batch_size,
                                      max_sen_len = opt.max_sen_len,max_doc_len=opt.max_doc_len,
                                      test=(x is 'test'))for x in ['train','test']}
        train_staticDataset=pairDataset(train,word_id_mapping,batchsize=opt.batch_size,max_sen_len=opt.max_sen_len,max_doc_len=opt.max_doc_len,test=True)
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True,drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)
        train_staticloader = DataLoader(train_staticDataset,shuffle=False)

        max_acc_subtask, max_f1 = [-1.] * 2
        for i in range(opt.training_iter):
            start_time, step = time.time(), 1
            # train
            for _, data in enumerate(trainloader):
                with torch.autograd.set_detect_anomaly(True):
                    x,sen_len, keep_prob1, keep_prob2,distance,true_y=data
                    pred_pair, reg=model(x,sen_len,opt.keep_prob1,opt.keep_prob2,distance)
                    pred_pair = pred_pair.cpu()
                    reg = reg.cpu()
                    loss = -torch.mean(true_y.float()*torch.log(pred_pair))+reg*opt.l2_reg
                    optimizer.zero_grad()
                    if use_gpu:
                        loss = loss.cuda()
                    loss.backward()
                    optimizer.step()
                    true_y_op = torch.argmax(true_y,1)
                    pred_y_op = torch.argmax(pred_pair,1)

                    acc = torch.mean(torch.eq(true_y_op,pred_y_op).float())
                    if use_gpu:
                        true_y_op = true_y_op.cpu()
                        pred_y_op = pred_y_op.cpu()
                        acc = acc.cpu()

                    print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss, acc))
                    step = step + 1

            # test
            testloss = 0
            testdatanum = 0
            alltestpred_y = torch.tensor([])
            alltesty = torch.tensor([])
            alldoc_len = torch.tensor([])
            allsen_len = torch.tensor([])
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x,sen_len, keep_prob1, keep_prob2,distance,true_y=data
                    pred_y,reg = model(x,sen_len,1,1,distance)
                    if use_gpu:
                        pred_y = pred_y.cpu()
                        reg = reg.cpu()
                    alltestpred_y = torch.cat((alltestpred_y,pred_y.float()),0)
                    alltesty = torch.cat((alltesty,true_y.float()),0)
                loss = -torch.mean(alltesty.float()*torch.log(alltestpred_y))+reg*opt.l2_reg
                alltesty_op = torch.argmax(alltesty,1)
                alltestpred_y_op = torch.argmax(alltestpred_y,1)
                acc = torch.mean(torch.eq(alltesty_op,alltestpred_y_op).float())
                if use_gpu:
                    alltesty_op = alltestpred_y.cpu()
                    alltestpred_y_op = alltestpred_y_op.cpu()
                    acc = acc.cpu()
                    loss = loss.cpu()

            print('\nepoch {}: test loss {:.4f}, acc {:.4f}, cost time: {:.1f}s\n'.format(i, loss, acc,
                                                                                          time.time() - start_time))
            if acc > max_acc_subtask:
                max_acc_subtask = acc
            print('max_acc_subtask: {:.4f} \n'.format(max_acc_subtask))

            # p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y, fold, save_dir)
            p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(NLP_Dataset["test"].pair_id_all, NLP_Dataset["test"].pair_id, alltestpred_y_op)
            if f1 > max_f1:
                max_keep_rate, max_p, max_r, max_f1 = keep_rate, p, r, f1
            print('original o_p {:.4f} o_r {:.4f} o_f1 {:.4f}'.format(o_p, o_r, o_f1))
            print('pair filter keep rate: {}'.format(keep_rate))
            print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))

            print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p, max_r, max_f1))

        print('Optimization Finished!\n')
        print('############# fold {} end ###############'.format(fold))
        # fold += 1
        acc_subtask_list.append(max_acc_subtask)
        keep_rate_list.append(max_keep_rate)
        p_pair_list.append(max_p)
        r_pair_list.append(max_r)
        f1_pair_list.append(max_f1)
        o_p_pair_list.append(o_p)
        o_r_pair_list.append(o_r)
        o_f1_pair_list.append(o_f1)

    print_training_info()
    all_results = [acc_subtask_list, keep_rate_list, p_pair_list, r_pair_list, f1_pair_list, o_p_pair_list,
                   o_r_pair_list, o_f1_pair_list]
    acc_subtask, keep_rate, p_pair, r_pair, f1_pair, o_p_pair, o_r_pair, o_f1_pair = map(
        lambda x: np.array(x).mean(), all_results)
    print('\nOriginal pair_predict: test f1 in 10 fold: {}'.format(np.array(o_f1_pair_list).reshape(-1, 1)))
    print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(o_p_pair, o_r_pair, o_f1_pair))
    print('\nAverage keep_rate: {:.4f}\n'.format(keep_rate))
    print('\nFiltered pair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1, 1)))
    print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))
    print_time()



if __name__ == '__main__':
    opt.scope='RNN'
    run()