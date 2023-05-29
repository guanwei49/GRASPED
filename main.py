import os

# import mlflow
from pathlib import Path


import torch
from torch.utils.data import DataLoader
from model.detect import detect
from model.train import train
from dataset import Dataset
import torch.utils.data as Data




def main(dataset,batch_size=64,n_epochs=20 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None ,enc_hidden_dim = 64 , encoder_num_layers = 4,decoder_num_layers=2, dec_hidden_dim = 64):
    '''
    :param dataset: instance of Dataset
    :param batch_size: size of mini batch
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim:  hidden dimension of GRU in encoder
    :param encoder_num_layers:  layers of GRU in encoder
    :param decoder_num_layers:  layers of GRU in decoder
    :param dec_hidden_dim: hidden dimension of GRU in decoder
    :return:  trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
    '''


    Xs=[]
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append( torch.LongTensor(dataset.features[i]))
    mask=torch.BoolTensor(dataset.mask)
    tensorDataset = Data.TensorDataset(*Xs,  mask)
    dataloader = DataLoader(tensorDataset, batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True, drop_last=True)

    gru_ae = train(dataloader,dataset.attribute_dims,n_epochs ,lr ,b1 ,b2 ,seed ,enc_hidden_dim  , encoder_num_layers ,decoder_num_layers, dec_hidden_dim )


    detect_dataloader = DataLoader(tensorDataset, batch_size=batch_size,
                            shuffle=False,num_workers=8,pin_memory=True)
    #
    attr_Shape=(dataset.num_cases,dataset.max_len,dataset.num_attributes)
    trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores = detect(gru_ae, detect_dataloader, dataset.attribute_dims)

    return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores



if __name__ == '__main__':
    attr_keys = ['concept:name', 'org:resource', 'org:role']
    threshold = 0.95

    ROOT_DIR = Path(__file__).parent
    logPath = os.path.join(ROOT_DIR, 'BPIC20_PrepaidTravelCost.xes')
    dataset = Dataset(logPath,attr_keys)
    trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset, batch_size=64,
                                                                                                n_epochs=20, lr=0.0002,
                                                                                                encoder_num_layers=4,
                                                                                                decoder_num_layers=2,
                                                                                                enc_hidden_dim=64,
                                                                                                dec_hidden_dim=64)
    attr_level_detection =(attr_level_abnormal_scores>threshold).astype('int64')
    event_level_detection =((attr_level_abnormal_scores>threshold).sum(axis=2)>=1).astype('int64')
    trace_level_detection = ((attr_level_abnormal_scores > threshold).sum(axis=(1,2))>=1).astype('int64')


    print(attr_level_detection)
    print(event_level_detection)
    print(trace_level_detection)

