import numpy as np
import torch
from tqdm import tqdm

from model import device


def detect(gru_ae, dataloader, attribute_dims):
    gru_ae.eval()
    with torch.no_grad():
        print("*" * 10 + "detecting" + "*" * 10)
        final_res = []
        for Xs in tqdm(dataloader):
            mask = Xs[-1]
            Xs = Xs[:-1]

            mask=mask.to(device)
            for k,tempX in enumerate(Xs):
                Xs[k] = tempX.to(device)

            fake_X = gru_ae(Xs)

            for attr_index in range(len(attribute_dims)):
                fake_X[attr_index]=torch.softmax(fake_X[attr_index],dim=2)

            this_res = []
            for attr_index in range(len(attribute_dims)):
                temp = fake_X[attr_index]
                index = Xs[attr_index].unsqueeze(2)
                probs= temp.gather(2, index)
                temp[(temp <= probs)] = 0
                res=temp.sum(2)
                res=res*(~mask)
                this_res.append(res)

            final_res.append(torch.stack(this_res,2))

        attr_level_abnormal_scores = np.array(torch.concatenate(final_res, 0).detach().cpu())
        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
