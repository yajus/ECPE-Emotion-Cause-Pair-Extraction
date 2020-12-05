import time
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合

    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))  # 每个词的位置及词

    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd  # 每个字的vector 1*200

    embedding = [list(np.zeros(embedding_dim))]  # 200
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def get_weight_varible(shape):
    w =(-0.01 - 0.01) * torch.rand(shape) + 0.01

    # shape = torch.empty(shape)
    return w

def getmask(length, max_len, out_shape,use_gpu):
    '''
    length shape:[batch_size]
    '''
    a=torch.zeros([len(length),max_len])
    if use_gpu:
        a=a.cuda()
    for i in range(len(length)):
        a[i,0:length[i]]=1
    ret = a.float()
    return torch.reshape(ret, out_shape)

def softmax_by_length(inputs, length,use_gpu):
    '''
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    '''
    expinputs = torch.exp(inputs.float())
    # print('before')
    # print(inputs)
    inputs = expinputs*getmask(length, expinputs.shape[2], expinputs.shape,use_gpu)
    # print('after')
    # print(inputs)
    _sum = torch.sum(inputs,  dim=2, keepdim =True) + 1e-9
    return inputs / _sum

def att_var(inputs,length,w1,b1,w2,use_gpu):#今后不用该函数
    '''
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (inputs.shape[1], inputs.shape[2])
    tmp = torch.reshape(inputs,[-1, n_hidden])
    u = torch.tanh(torch.matmul(tmp, w1) + b1)
    alpha = torch.reshape(torch.matmul(u, w2), [-1, 1, max_len])#200->1
    alpha = softmax_by_length(alpha, length,use_gpu)
    return torch.reshape(torch.matmul(alpha, inputs), [-1, n_hidden])

def acc_prf(pred_y, true_y, doc_len, average='binary'):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1


def prf_2nd_step(pair_id_all, pair_id, pred_y, fold=0, save_dir=''):
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])

    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir + 'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0] / 10000, pair_id_all[-1] / 10000
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e + 1):
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id / 10000, p_id % 10000 / 100, p_id % 100
                if d != doc_id: break
                true_pair.append((p1, p2))
                line += '({}, {}) '.format(p1, p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all): break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id / 10000, p_id % 10000 / 100, p_id % 100
                if d != doc_id: break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair) > 1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)

    if fold:
        write_log()
    keep_rate = len(pair_id_filtered) / (len(pair_id) + 1e-8)
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)
    o_acc_num = len(s1 & s2)
    acc_num = len(s1 & s3)
    o_p, o_r = o_acc_num / (len(s2) + 1e-8), o_acc_num / (len(s1) + 1e-8)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1, o_f1 = 2 * p * r / (p + r + 1e-8), 2 * o_p * o_r / (o_p + o_r + 1e-8)

    return p, r, f1, o_p, o_r, o_f1, keep_rate