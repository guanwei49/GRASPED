import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from model import device
from model.GRU_AE import  GRU_AE

def train(dataloader,attribute_dims,n_epochs ,lr ,b1 ,b2 ,seed,enc_hidden_dim , encoder_num_layers ,decoder_num_layers, dec_hidden_dim):

    if type(seed) is int:
        torch.manual_seed(seed)

    gru_ae = GRU_AE(attribute_dims, enc_hidden_dim, encoder_num_layers,decoder_num_layers ,dec_hidden_dim)


    gru_ae.to(device)

    optimizer = torch.optim.Adam(gru_ae.parameters(),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            mask=Xs[-1]
            mask=mask.to(device)
            Xs = Xs[:-1]
            for k ,X in enumerate(Xs):
                Xs[k]=X.to(device)

            fake_X = gru_ae(Xs,mask)

            optimizer.zero_grad()

            loss=0.0
            for ij in range(len(attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                pred = torch.softmax(fake_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                true = Xs[ij][:, 1:].flatten()

                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                               fake_X[0].shape[1] - 1)

                cross_entropys = -torch.log(corr_pred)
                loss += cross_entropys.masked_select((~mask[:, 1:])).mean()


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

