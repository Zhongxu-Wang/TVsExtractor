import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from datasetEMA import Datasetsys
from tqdm import tqdm
from model.EMA_Predictor import EMA_Predictor
from tools import to_device

def log_norm(x, mean=-4, std=4, dim=2):
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x
with open("LibriTTS_dataset.txt", 'r', encoding='UTF-8') as f:
    Data_list = f.readlines()
Data_list = [l.split('|')[0] for l in Data_list]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):

    model = EMA_Predictor().to(device)
    dataset = Datasetsys(batch_size=train_config["optimizer"]["batch_size"], sort=False, drop_last=False, file_list = Data_list)
    loader = DataLoader(dataset, batch_size=train_config["optimizer"]["batch_size"], shuffle=False, collate_fn=dataset.collate_fn, num_workers=8)
    ckpt = torch.load(os.path.join(train_config["path"]["ckpt_path"], "300000.pth.tar"))
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.requires_grad_ = False

    for batchs in tqdm(loader):
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                [EMAs, mel_input_length, F0, energy] = model(*(batch[1:]), synthesize = True)
                EMAs = EMAs.detach().cpu().numpy()
                F0 = F0.squeeze().detach().cpu().numpy()
                energy = energy.squeeze().detach().cpu().numpy()

                for i, name in enumerate(batch[0]):
                    speaker = name.split("/")[-1].split(".")[0]
                    np.save(os.path.join(args.saving_path,"EMA", speaker),EMAs[i,:mel_input_length[i],:],)
                    np.save(os.path.join(args.saving_path,"F0", speaker),F0[i,:mel_input_length[i]],)
                    np.save(os.path.join(args.saving_path,"energy", speaker),energy[i,:mel_input_length[i]],)
                del(EMAs)
                del(F0)
                del(energy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--saving_path", type=str, default="../111")
    args = parser.parse_args()

    train_config = yaml.load(open("config/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (train_config)

    os.makedirs(args.saving_path+"/EMA", exist_ok=True)
    os.makedirs(args.saving_path+"/F0", exist_ok=True)
    os.makedirs(args.saving_path+"/energy", exist_ok=True)
    main(args, configs)
