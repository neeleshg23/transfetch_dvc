import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

import dvc.api
from dvclive import Live

from models.v import TMAP
from models.d import DenseNetTeacher
from models.r import resnet50

torch.manual_seed(100)

model = None
optimizer = None
scheduler = None
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        output = sigmoid(model(data))
        loss = F.binary_cross_entropy(output, target, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = sigmoid(model(data))
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, cur_tch, loading, model_save_path, train_loader, test_loader, live):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        print("-------------Model Loaded------------")
        
    best_loss=0
    early_stop = early_stop
    curr_early_stop = early_stop

    for epoch in range(epochs):
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        print((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        live.log_metric(f"train_tch_{cur_tch}/train_loss", train_loss)
        live.log_metric(f"train_tch_{cur_tch}/test_loss", test_loss)

        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            print("-------- Save Best Model! --------")
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            print("Early Stop Left: {}".format(curr_early_stop))
        if curr_early_stop == 0:
            print("-------- Early Stop! --------")
            break

        live.next_step()

    
def main():
    global model
    global optimizer
    global scheduler

    params = dvc.api.params_show("f.yaml")

    if len(sys.argv) != 2:
        print("Current teacher number 1 - num_tch is required")
        return
    cur_tch = sys.argv[1]

    option = params["teacher"][f"model_{cur_tch}"]

    gpu_id = params["system"]["gpu-id"]
    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]

    if option == "d":
        channels = params["model"][f"tch_{option}"]["channels"]
        model = DenseNetTeacher(num_classes, channels)
    elif option == "r":
        dim = params["model"][f"tch_{option}"]["dim"]
        channels = params["model"][f"tch_{option}"]["channels"]
        model = resnet50(num_classes, channels)
    elif option == "v":
        dim = params["model"][f"tch_{option}"]["dim"]
        depth = params["model"][f"tch_{option}"]["depth"]
        heads = params["model"][f"tch_{option}"]["heads"]
        mlp_dim = params["model"][f"tch_{option}"]["mlp-dim"]
        channels = params["model"][f"tch_{option}"]["channels"]
        model = TMAP(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=mlp_dim
        )
    # elif option == "m":
    #     model = tch_m
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    model_save_path = os.path.join(model_dir, f"teacher_{cur_tch}.pth")

    train_loader = torch.load(os.path.join(processed_dir, f"train_loader_{cur_tch}"))
    test_loader = torch.load(os.path.join(processed_dir, f"test_loader_{cur_tch}"))

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loading = False

    with Live(dir="res", resume=True) as live:
        live.step = 1
        run_epoch(epochs, early_stop, cur_tch, loading, model_save_path, train_loader, test_loader, live)

if __name__ == "__main__":
    main()