import torch
batch_size=2
hidden_size=5
embedding_dim=6
seq_length=2
num_layers=1
num_directions=1
vocab_size=20
import numpy as np
input_data=np.random.uniform(0,19,size=(batch_size,seq_length))
b=np.random.uniform(0,19,size=(batch_size,seq_length))
input_data=[np.append(input_data,b,axis=0)]
input_data = np.append(input_data,input_data,axis=0)
input_data = torch.FloatTensor(input_data)
length = [3,2]
a=torch.zeros([len(length),4,2])
for i in range(len(
        length)):
    a[i,0:length[i],:]=1
ret = a.float()
ret = torch.FloatTensor(ret)
input_data = input_data*a

# input_data=torch.from_numpy(input_data).long()
# embedding_layer=torch.nn.Embedding(vocab_size,embedding_dim)
print(input_data)
# lstm_layer=torch.nn.LSTM(input_size=seq_length,hidden_size=hidden_size,num_layers=num_layers,
#                         bias=True,batch_first=True,bidirectional=True)
# output,(h_n,c_n)=lstm_layer(torch.FloatTensor(input_data))
# print(output.shape)
# print(output)
