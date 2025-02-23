import os
import yaml
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ScheduledOptim
from model.EMA_Predictor import EMA_Predictor
from datasetEMA import DatasetEMA
from tools import to_device, log
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    print("Prepare training ...")
    train_config = configs
    all_data = os.listdir(train_config["path"]["preprocessed_path"])

    DataList = [i for i in all_data if i.split("_")[0]]
    Val_list = random.sample(DataList, 100)
    Train_List = list(set(DataList) - set(Val_list))

    Train_dataset = DatasetEMA(train_config, sort=True, drop_last=True, file_list = Train_List)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    Train_loader = DataLoader(Train_dataset, batch_size=batch_size * group_size, shuffle=True, collate_fn=Train_dataset.collate_fn)
    Val_dataset = DatasetEMA(train_config, sort=False, drop_last=False, file_list = Val_list)
    Val_loader = DataLoader(Val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Val_dataset.collate_fn)

    #Prepare model
    model = EMA_Predictor().to(device)
    optimizer = ScheduledOptim(model, train_config, args.restore_step)
    if args.restore_step:
        ckpt = torch.load(os.path.join(train_config["path"]["ckpt_path"], str(args.restore_step)+".pth.tar"))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    model.train()
    model = nn.DataParallel(model)

    Loss = EMALoss().to(device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    Time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    train_log_path = os.path.join(train_config["path"]["log_path"], Time)
    os.makedirs(train_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(Train_loader), desc="Epoch {}".format(epoch), position=1)
        for Tbatchs in Train_loader:
            for Tbatch in Tbatchs:
                Tbatch = to_device(Tbatch, device)
                Toutput = model(*(Tbatch[1:]))
                losses = Loss(Tbatch, Toutput)
                total_loss = losses[0]
                total_loss.backward()
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
                if step % log_step == 0:
                    # losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, relative_error: {:.4f},corr: {:.4f}".format(*losses)
                    with open(os.path.join(train_log_path, "train_log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig = Plot(Tbatch[-2], Toutput[0])
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, Tbatch[0][0]))

                if step % val_step == 0:
                    model.eval()
                    sum_loss = 0
                    rel_loss = 0
                    corr = 0
                    for Vbatchs in Val_loader:
                        for Vbatch in Vbatchs:
                            Vbatch = to_device(Vbatch, device)
                            with torch.no_grad():
                                Voutput = model(*(Vbatch[1:]))
                                Vlosses = Loss(Vbatch, Voutput)
                                sum_loss += Vlosses[0].item() * len(Vbatch[0])
                                rel_loss += Vlosses[1].item() * len(Vbatch[0])
                                corr += Vlosses[2].item() * len(Vbatch[0])
                    sum_loss = sum_loss / len(Val_dataset)
                    rel_loss = rel_loss / len(Val_dataset)
                    corr = corr / len(Val_dataset)
                    message = "Validation Step {}, Val Loss: {:.4f}, rel Loss: {:.4f}".format(
                        *([step] + [sum_loss] + [rel_loss]))
                    fig = Plot(Vbatch[-2], Voutput[0])
                    outer_bar.write(message)
                    log(train_logger,fig=fig,
                        tag="Valing/step_{}_{}".format(step, Vbatch[0][0]))
                    train_logger.add_scalar("val/Val_loss", sum_loss, step)
                    train_logger.add_scalar("val/rel_loss", rel_loss, step)
                    train_logger.add_scalar("val/corr", corr, step)
                    model.train()

                if step == total_step:
                    torch.save({"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict()},
                        os.path.join(train_config["path"]["ckpt_path"],"{}.pth.tar".format(step)))
                    quit()
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1

def Plot(gt, pre):
    gt = gt.detach().cpu().numpy()
    pre1 = pre.detach().cpu().numpy()
    fig, axs = plt.subplots(10, 1, facecolor='white')
    for num, ax in zip(range(10), axs):
        ax.plot(gt[0,:,num], 'b-')
        ax.plot(pre1[0,:,num], 'r-')
    return fig

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class EMALoss(nn.Module):
    def __init__(self,):
        super(EMALoss, self).__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        EMA_targets = inputs[-2]
        [EMA,mel_masks] = predictions
        mel_masks = ~mel_masks
        EMA_targets.requires_grad = False
        EMA = EMA.masked_select(mel_masks)
        EMA_targets = EMA_targets.masked_select(mel_masks)
        loss1 = self.mae_loss(EMA_targets, EMA)
        loss2 = ((EMA - EMA_targets).abs() / EMA_targets.abs()).mean()
        corr=np.corrcoef(EMA.detach().cpu().numpy(),EMA_targets.detach().cpu().numpy())
        return [loss1, loss2, corr[0][1]]

if __name__ == "__main__":
    setup_seed(2023)
    parser = argparse.ArgumentParser()

    parser.add_argument("-r","--restore_step", type=int, default=0)
    args = parser.parse_args()

    # Read Config
    train_config = yaml.load(open("config/train.yaml", "rb"), Loader=yaml.FullLoader)

    main(args, train_config)
