import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from model import device
from model.GRU_AE import  GRU_AE

def train(dataloader,attribute_dims,n_epochs=30 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None,enc_hidden_dim = 50 , encoder_num_layers = 8,decoder_num_layers=4, dec_hidden_dim = 50):
    '''
    GRU_AE
    :param dataloader:
    :param attribute_dims:  Number of attribute values per attribute : list
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim: encoder hidden dimensions :GRU
    :param encoder_num_layers: Number of encoder layers :GRU
    :param dec_hidden_dim:  decoder hidden dimensions :GRU
    :param teacher_forcing_ratio:
    :return: gru_ae
    '''

    if type(seed) is int:
        torch.manual_seed(seed)

    gru_ae = GRU_AE(attribute_dims, enc_hidden_dim, encoder_num_layers,decoder_num_layers ,dec_hidden_dim)

    loss_func = nn.CrossEntropyLoss()

    gru_ae.to(device)

    optimizer = torch.optim.Adam(gru_ae.parameters(),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            Xs = Xs[:-1]
            for k ,X in enumerate(Xs):
                Xs[k]=X.to(device)

            fake_X = gru_ae(Xs)

            optimizer.zero_grad()

            loss=0.0
            for ij in range(len(attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                pred=fake_X[ij][:,1:,:].flatten(0,-2)
                true=Xs[ij][:,1:].flatten()
                loss+=loss_func(pred,true)

            train_loss += loss.item() * Xs[0].shape[0]
            train_num +=Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return gru_ae

