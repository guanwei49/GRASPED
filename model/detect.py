import numpy as np
import torch
from tqdm import tqdm

from model import device


def detect(gru_ae, dataloader, attribute_dims, attr_Shape):
    gru_ae.eval()
    with torch.no_grad():
        attr_level_abnormal_scores=np.zeros(attr_Shape)
        print("*" * 10 + "detecting" + "*" * 10)
        for index, Xs in enumerate(tqdm(dataloader)):
            case_len = Xs[-1]
            Xs = Xs[:-1]
            for k,tempX in enumerate(Xs):
                Xs[k] = tempX.to(device)

            fake_X = gru_ae(Xs)

            for k in range(len(fake_X)):
                fake_X[k] = fake_X[k].detach().cpu()
            for attr_index in range(len(attribute_dims)):
                fake_X[attr_index]=torch.softmax(fake_X[attr_index],dim=2)


            for batch_i in range(len(case_len)):
                for time_step in range(1,case_len[batch_i]):
                    for attr_index in range(len(attribute_dims)):
                        # 取比实际出现的属性值大的其他属性值的概率之和
                        truepos=Xs[attr_index][batch_i,time_step]
                        attr_level_abnormal_scores[index*len(case_len)+batch_i,time_step,attr_index]=fake_X[attr_index][batch_i,time_step][fake_X[attr_index][batch_i,time_step]>fake_X[attr_index][batch_i,time_step,truepos]].sum()
        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
