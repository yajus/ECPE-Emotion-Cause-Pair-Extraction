import argparse
import torch
import torch.nn
from torch import nn
import torchvision
import os, sys
from data_process import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.utils.multiclass import type_of_target

#### setting agrparse
parser = argparse.ArgumentParser(description='Training')
##### input
parser.add_argument('--w2v_file', type=str, default='../data/w2v_200.txt', help='embedding file')
parser.add_argument('--embedding_dim', type=int, default=200, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, help='dimension of position embedding')
parser.add_argument('--max_sen_len', type=int, default=30, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=75, help='max number of tokens per documents')
##### model struct
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
##### traing
parser.add_argument('--training_iter', type=int, default=15, help='number of train iterator')
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
    use_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        opt.batch_size, opt.learning_rate, opt.keep_prob1, opt.keep_prob2, opt.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(opt.training_iter, opt.scope))


class MyDataset(Dataset):
    def __init__(self, input_file, word_idx, batchsize=opt.batch_size, max_doc_len=75, max_sen_len=45, test=False,
                 transforms=None):
        self.transforms = transforms  # NLP中实际上没用到
        print('load data_file: {}'.format(input_file))
        self.y_pairs, self.doc_len, self.y_position, self.y_cause, self.x, self.sen_len = [], [], [], [], [], []
        self.doc_id = []
        self.test = test
        self.n_cut = 0
        self.batch_size = batchsize
        inputFile = open(input_file, 'r')
        while True:
            line = inputFile.readline()
            if line == '': break
            line = line.strip().split()
            self.doc_id.append(line[0])
            d_len = int(line[1])
            pairs = eval('[' + inputFile.readline().strip() + ']')
            self.doc_len.append(d_len)
            self.y_pairs.append(pairs)
            pos, cause = zip(*pairs)
            y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(
                max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
            for i in range(d_len):
                y_po[i][int(i + 1 in pos)] = 1
                y_ca[i][int(i + 1 in cause)] = 1  ####没有保留配对信息
                words = inputFile.readline().strip().split(',')[-1]
                sen_len_tmp[i] = min(len(words.split()), max_sen_len)
                for j, word in enumerate(words.split()):
                    if j >= max_sen_len:
                        self.n_cut = self.n_cut + 1
                        break
                    x_tmp[i][j] = int(word_idx[word])

            self.y_position.append(y_po)
            self.y_cause.append(y_ca)
            self.x.append(x_tmp)
            self.sen_len.append(sen_len_tmp)

        self.y_position, self.y_cause, self.x, self.sen_len, self.doc_len = map(np.array,
                                                                                [self.y_position, self.y_cause, self.x,
                                                                                 self.sen_len, self.doc_len])
        for var in ['self.y_position', 'self.y_cause', 'self.x', 'self.sen_len', 'self.doc_len']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('n_cut {}'.format(self.n_cut))
        print('load data done!\n')

        self.index = [i for i in range(len(self.y_cause))]

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x[index], self.sen_len[index], self.doc_len[index], opt.keep_prob1, opt.keep_prob2,
                     self.y_position[index],
                     self.y_cause[index]]
        return feed_list

    def __len__(self):
        return len(self.x)


class attention(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(attention, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.sh2 = 2 * opt.n_hidden
        # self.linearlayer1 = nn.Linear(self.sh2,self.sh2).cuda()
        # self.linearlayer2 = nn.Linear(self.sh2,1,bias=False).cuda()
        self.w1 = get_weight_varible([self.sh2, self.sh2])
        self.w1.requires_grad = True
        self.b1 = get_weight_varible([self.sh2])
        self.b1.requires_grad = True
        self.w2 = get_weight_varible([self.sh2, 1])
        self.w2.requires_grad = True
        if use_gpu:
            self.w1 = self.w1.cuda()
            self.b1 = self.b1.cuda()
            self.w2 = self.w2.cuda()

    def forward(self, inputs, sen_len):
        r_out, (h_n, h_c) = self.bilstm(inputs)
        inputs = r_out
        self.sen_len = sen_len
        max_len, n_hidden = (inputs.shape[1], inputs.shape[2])
        tmp = torch.reshape(inputs, [-1, n_hidden])
        u = torch.tanh(torch.matmul(tmp, self.w1) + self.b1)
        alpha = torch.reshape(torch.matmul(u, self.w2), [-1, 1, max_len])  # 200->1
        alpha = softmax_by_length(alpha, self.sen_len, use_gpu)
        s = torch.reshape(torch.matmul(alpha, inputs), [-1, n_hidden])
        s = torch.reshape(s, [-1, opt.max_doc_len, 2 * opt.n_hidden])
        return s


class biLSTM(torch.nn.Module):
    def __init__(self, word_embedding, keep_prob1, keep_prob2, input_size=opt.n_hidden * 2, hidden_size=opt.n_hidden):
        super(biLSTM, self).__init__()

        self.causebilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.posbilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.attention = attention(input_size=input_size, hidden_size=hidden_size)
        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2
        self.word_embedding = word_embedding
        self.w1 = get_weight_varible([2 * opt.n_hidden, opt.n_class])
        self.w1.requires_grad = True
        self.b1 = get_weight_varible([opt.n_class])
        self.b1.requires_grad = True
        self.w2 = get_weight_varible([2 * opt.n_hidden, opt.n_class])
        self.w2.requires_grad = True
        self.b2 = get_weight_varible([opt.n_class])
        self.b2.requires_grad = True
        if use_gpu:
            self.w1 = self.w1.cuda()
            self.w2 = self.w2.cuda()
            self.b1 = self.b1.cuda()
            self.b2 = self.b2.cuda()

        # self.nnlayer_cause = nn.Linear(2 * opt.n_hidden, opt.n_class).cuda()
        # self.nnlayer_pos = nn.Linear(2 * opt.n_hidden, opt.n_class).cuda()

    def forward(self, x, sen_len, doc_len, keep_prob1, keep_prob2):
        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2
        self.sen_len = sen_len
        self.doc_len = doc_len
        x = x.long()
        x = torch.index_select(self.word_embedding, dim=0, index=torch.reshape(x, [-1]))
        if use_gpu:
            x = x.cuda()
        # x = x.float()
        # print('输出尺寸')
        # print(x.shape)
        # 输出尺寸
        # torch.Size([72000, 200])
        inputs = torch.reshape(x, [-1, opt.max_sen_len, opt.embedding_dim])
        inputs = torch.nn.Dropout(1.0 - self.keep_prob1)(inputs)
        sen_len = torch.reshape(self.sen_len, [-1])
        s = self.attention(inputs, sen_len)
        r_out, (h_n, h_c) = self.causebilstm(s)
        s = r_out
        s1 = torch.reshape(s, [-1, 2 * opt.n_hidden])
        s1 = torch.nn.Dropout(1.0 - self.keep_prob2)(s1)
        pred_cause = F.softmax(torch.matmul(s1, self.w1) + self.b1)
        pred_cause = torch.reshape(pred_cause, [-1, opt.max_doc_len, opt.n_class])

        ss = self.attention(inputs, sen_len)
        r_out, (h_n, h_c) = self.posbilstm(ss)
        ss = r_out
        ss1 = torch.reshape(ss, [-1, 2 * opt.n_hidden])
        ss1 = torch.nn.Dropout(1.0 - self.keep_prob2)(ss1)
        pred_pos = F.softmax(torch.matmul(ss1, self.w2) + self.b2)
        pred_pos = torch.reshape(pred_pos, [-1, opt.max_doc_len, opt.n_class])

        reg = torch.norm(self.w1) + torch.norm(self.b1)
        reg = reg + (torch.norm(self.w2) + torch.norm(self.b2))
        reg = reg / 2
        return pred_pos, pred_cause, reg


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

    # wordembedding
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(opt.embedding_dim, opt.embedding_dim_pos,
                                                                            'data_combine/clause_keywords.csv',
                                                                            opt.w2v_file)
    word_embedding = torch.FloatTensor(word_embedding)
    pos_embedding = torch.FloatTensor(pos_embedding)

    # train
    print_training_info()
    acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
    acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
    p_pair_list, r_pair_list, f1_pair_list = [], [], []
    for fold in range(1, 11):
        # model
        print('build model..')
        model = biLSTM(word_embedding, opt.keep_prob1, opt.keep_prob2, opt.n_hidden * 2, opt.n_hidden)
        print('build model end...')
        if use_gpu:
            model = model.cuda()

        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        print('############# fold {} begin ###############'.format(fold))
        train = 'data_combine/' + train_file_name
        test = 'data_combine/' + test_file_name
        edict = {"train": train, "test": test}
        NLP_Dataset = {x: MyDataset(edict[x], word_id_mapping, batchsize=opt.batch_size
                                    , max_sen_len=opt.max_sen_len, max_doc_len=opt.max_doc_len
                                    , test=(x is 'test')) for x in ['train', 'test']}
        train_staticDataset = MyDataset(train, word_id_mapping, batchsize=opt.batch_size, max_sen_len=opt.max_sen_len,
                                        max_doc_len=opt.max_doc_len, test=True)
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)
        train_staticloader = DataLoader(train_staticDataset, shuffle=False)

        max_f1_cause, max_f1_pos, max_f1_avg = [-1.] * 3
        for i in range(opt.training_iter):
            start_time, step = time.time(), 1
            for _, data in enumerate(trainloader):
                with torch.autograd.set_detect_anomaly(True):
                    x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause = data
                    pred_pos, pred_cause, reg = model(x, sen_len, doc_len, opt.keep_prob1, opt.keep_prob2)
                    pred_pos = pred_pos.cpu()
                    pred_cause = pred_cause.cpu()
                    reg = reg.cpu()
                    # tensor([17., 13.,  9., 18., 16., 10., 13.,  7., 24., 13., 17., 13., 12., 14.,
                    #         13., 14., 12., 11., 15., 10., 15., 20., 12., 25., 20., 18., 32., 10.,
                    #         30., 15.,  9., 20.])
                    valid_num = torch.sum(doc_len)
                    loss_pos = - torch.sum(y_position * torch.log(pred_pos).double()) / valid_num
                    loss_cause = - torch.sum(y_cause * torch.log(pred_cause).double()) / valid_num
                    loss_op = loss_cause * opt.cause + loss_pos * opt.pos + reg.double() * opt.l2_reg
                    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
                    optimizer.zero_grad()
                    if use_gpu:
                        loss_op = loss_op.cuda()
                    loss_op.backward()
                    optimizer.step()

                    true_y_cause_op = torch.argmax(y_cause, 2)
                    pred_y_cause_op = torch.argmax(pred_cause, 2)
                    true_y_pos_op = torch.argmax(y_position, 2)
                    pred_y_pos_op = torch.argmax(pred_pos, 2)
                    # print(y_cause.shape)
                    if use_gpu:
                        true_y_cause_op = true_y_cause_op.cpu()
                        pred_y_cause_op = pred_y_cause_op.cpu()
                        true_y_pos_op = true_y_pos_op.cpu()
                        pred_y_pos_op = pred_y_pos_op.cpu()

                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss_op))
                        acc, p, r, f1 = acc_prf(pred_y_cause_op, true_y_cause_op, doc_len)
                        print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                        acc, p, r, f1 = acc_prf(pred_y_pos_op, true_y_pos_op, doc_len)
                        print('position_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                    step = step + 1
                    testloss = 0
                    testdatanum = 0
                    alltestpred_pos = torch.tensor([])
                    alltestpred_cause = torch.tensor([])
                    alltesty_cause = torch.tensor([])
                    alltesty_pos = torch.tensor([])
                    alldoc_len = torch.tensor([])
                    allsen_len = torch.tensor([])

            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause = data
                    pred_pos, pred_cause, reg = model(x, sen_len, doc_len, 1, 1)
                    if use_gpu:
                        pred_pos = pred_pos.cpu()
                        pred_cause = pred_cause.cpu()
                        reg = reg.cpu()
                    alltestpred_cause = torch.cat((alltestpred_cause, pred_cause.float()), 0)
                    alltestpred_pos = torch.cat((alltestpred_pos, pred_pos.float()), 0)
                    alltesty_pos = torch.cat((alltesty_pos, y_position.float()), 0)
                    alltesty_cause = torch.cat((alltesty_cause, y_cause.float()), 0)
                    alldoc_len = torch.cat((alldoc_len, doc_len.float()), 0)
                    allsen_len = torch.cat((allsen_len, sen_len.float()), 0)
                    valid_num = torch.sum(doc_len).float()
                    testdatanum = testdatanum + valid_num

                loss_pos = -torch.sum(alltesty_pos * torch.log(alltestpred_pos)) / testdatanum
                loss_cause = - torch.sum(alltesty_cause * torch.log(alltestpred_cause)) / testdatanum
                loss = loss_cause * opt.cause + loss_pos * opt.pos + reg * opt.l2_reg
                print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time() - start_time))

                alltesty_cause_op = torch.argmax(alltesty_cause, 2)
                alltestpred_cause_op = torch.argmax(alltestpred_cause, 2)
                alltesty_pos_op = torch.argmax(alltesty_pos, 2)
                alltestpred_pos_op = torch.argmax(alltestpred_pos, 2)
                acc, p, r, f1 = acc_prf(alltestpred_cause_op.numpy(), alltesty_cause_op.numpy(),
                                        alldoc_len.int().numpy())
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('cause_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_cause, max_p_cause,
                                                                                        max_r_cause, max_f1_cause))

                acc, p, r, f1 = acc_prf(alltestpred_pos_op.numpy(), alltesty_pos_op.numpy(), alldoc_len.int().numpy())
                result_avg_pos = [acc, p, r, f1]
                if f1 > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc, p, r, f1
                print('position_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                print(
                    'max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_pos, max_p_pos, max_r_pos,
                                                                                      max_f1_pos))

                if (result_avg_cause[-1] + result_avg_pos[-1]) / 2. > max_f1_avg:
                    max_f1_avg = (result_avg_cause[-1] + result_avg_pos[-1]) / 2.
                    result_avg_cause_max = result_avg_cause
                    result_avg_pos_max = result_avg_pos

                    te_pred_y_cause, te_pred_y_pos = alltestpred_cause_op, alltestpred_pos_op
                    tr_pred_y_cause, tr_pred_y_pos = [], []
                    for _, data in enumerate(train_staticloader):
                        x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause = data
                        pred_pos, pred_cause, reg = model(x, sen_len, doc_len, 1, 1)
                        if use_gpu:
                            pred_pos = pred_pos.cpu()
                            pred_cause = pred_cause.cpu()
                            reg = reg.cpu()
                        pred_cause = torch.argmax(pred_cause, 2)
                        pred_pos = torch.argmax(pred_pos, 2)
                        tr_pred_y_cause.extend(pred_cause.numpy())
                        tr_pred_y_pos.extend(pred_pos.numpy())
                print('Average max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(
                    result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2], result_avg_cause_max[3]))
                print('Average max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(
                    result_avg_pos_max[0], result_avg_pos_max[1], result_avg_pos_max[2], result_avg_pos_max[3]))

        def get_pair_data(file_name, doc_id, doc_len, y_pairs, pred_y_cause, pred_y_pos, x, sen_len, word_idx_rev):
            g = open(file_name, 'w')
            for i in range(len(doc_id)):
                g.write(doc_id[i] + ' ' + str(doc_len[i]) + '\n')
                g.write(str(y_pairs[i]) + '\n')
                for j in range(doc_len[i]):
                    clause = ''
                    for k in range(sen_len[i][j]):
                        clause = clause + word_idx_rev[x[i][j][k]] + ' '
                    g.write(str(j + 1) + ', ' + str(pred_y_pos[i][j]) + ', ' + str(
                        pred_y_cause[i][j]) + ', ' + clause + '\n')
            print('write {} done'.format(file_name))

        get_pair_data(save_dir + test_file_name, NLP_Dataset['test'].doc_id, NLP_Dataset['test'].doc_len,
                      NLP_Dataset['test'].y_pairs, te_pred_y_cause.numpy(),
                      te_pred_y_pos.numpy(), NLP_Dataset['test'].x, NLP_Dataset['test'].sen_len, word_idx_rev)
        get_pair_data(save_dir + train_file_name, train_staticDataset.doc_id, train_staticDataset.doc_len,
                      train_staticDataset.y_pairs, tr_pred_y_cause,
                      tr_pred_y_pos, train_staticDataset.x, train_staticDataset.sen_len, word_idx_rev)

        print('Optimization Finished!\n')
        print('############# fold {} end ###############'.format(fold))
        # fold += 1
        acc_cause_list.append(result_avg_cause_max[0])
        p_cause_list.append(result_avg_cause_max[1])
        r_cause_list.append(result_avg_cause_max[2])
        f1_cause_list.append(result_avg_cause_max[3])
        acc_pos_list.append(result_avg_pos_max[0])
        p_pos_list.append(result_avg_pos_max[1])
        r_pos_list.append(result_avg_pos_max[2])
        f1_pos_list.append(result_avg_pos_max[3])

    print_training_info()
    all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, acc_pos_list, p_pos_list,
                   r_pos_list, f1_pos_list]
    acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos = map(lambda x: np.array(x).mean(),
                                                                               all_results)
    print('\ncause_predict: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1, 1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
    print('position_predict: test f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1, 1)))
    print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
    print_time()


if __name__ == '__main__':
    opt.scope = 'Ind_BiLSTM_1'

    run()
