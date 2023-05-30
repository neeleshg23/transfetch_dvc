import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

import dvc.api
from dvclive import Live

from models.v import TMAP
from models.d import DenseNetStudent, DenseNetTeacher
from models.r import resnet50, resnet_tiny

torch.manual_seed(100)

model = None
teacher_1_model = None
teacher_2_model = None
teacher_3_model = None
teacher_4_model = None
optimizer = None
scheduler = None
device = None
alpha = None
Temperature = None

soft_loss = nn.KLDivLoss(reduction="mean", log_target=True)
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_preds = model(data)
        
        with torch.no_grad():
            teacher_1_preds = teacher_1_model(data)
            teacher_2_preds = teacher_2_model(data)
            teacher_3_preds = teacher_3_model(data)
            teacher_4_preds = teacher_4_model(data)

        student_loss = F.binary_cross_entropy(sigmoid(student_preds), target, reduction='mean')

        x_s_sig = sigmoid(student_preds / Temperature).reshape(-1)
        x_s_p = torch.stack((x_s_sig, 1 - x_s_sig), dim=1)

        x_t1_sig = sigmoid(teacher_1_preds / Temperature).reshape(-1)
        x_t1_p = torch.stack((x_t1_sig, 1 - x_t1_sig), dim=1)

        distillation_loss_1 = soft_loss(x_s_p.log(), x_t1_p.log())

        x_t2_sig = sigmoid(teacher_2_preds / Temperature).reshape(-1)
        x_t2_p = torch.stack((x_t2_sig, 1 - x_t2_sig), dim=1)

        distillation_loss_2 = soft_loss(x_s_p.log(), x_t2_p.log())

        x_t3_sig = sigmoid(teacher_3_preds / Temperature).reshape(-1)
        x_t3_p = torch.stack((x_t3_sig, 1 - x_t3_sig), dim=1)

        distillation_loss_3 = soft_loss(x_s_p.log(), x_t3_p.log())

        x_t4_sig = sigmoid(teacher_4_preds / Temperature).reshape(-1)
        x_t4_p = torch.stack((x_t4_sig, 1 - x_t4_sig), dim=1)

        distillation_loss_4 = soft_loss(x_s_p.log(), x_t4_p.log())

        loss = alpha * student_loss + (1 - alpha) * (distillation_loss_1 + distillation_loss_2 + distillation_loss_3 + distillation_loss_4)

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
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, loading, tch_1_model_save_path, tch_2_model_save_path, tch_3_model_save_path, tch_4_model_save_path, model_save_path, train_loader, test_loader, live):
    teacher_1_model.load_state_dict(torch.load(tch_1_model_save_path))
    print("-------------Teacher 1 Model Loaded------------")
    teacher_2_model.load_state_dict(torch.load(tch_2_model_save_path))
    print("-------------Teacher 2 Model Loaded------------")
    teacher_3_model.load_state_dict(torch.load(tch_3_model_save_path))
    print("-------------Teacher 3 Model Loaded------------")
    teacher_4_model.load_state_dict(torch.load(tch_4_model_save_path))
    print("-------------Teacher 3 Model Loaded------------")

    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        print("-------------Model Loaded------------")
        
    best_loss=0
    early_stop = early_stop
    curr_early_stop = early_stop

    for epoch in range(epochs):
        train_loss=train(epoch, train_loader, model_save_path)
        test_loss=test(test_loader)
        print((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        live.log_metric("train_stu/train_loss", train_loss)
        live.log_metric("train_stu/test_loss", test_loss)

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
    global teacher_1_model
    global teacher_2_model
    global teacher_3_model
    global teacher_4_model
    global optimizer
    global scheduler
    global device
    global alpha
    global Temperature

    params = dvc.api.params_show("f.yaml")

    tch_1_option = params["teacher"]["model_1"]
    tch_2_option = params["teacher"]["model_2"]
    tch_3_option = params["teacher"]["model_3"]
    tch_4_option = params["teacher"]["model_4"]

    stu_option = params["student"]["model"]

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
    alpha = params["train"]["alpha"]
    Temperature = params["train"]["temperature"]

    if tch_1_option == "d":
        channels = params["model"][f"tch_{tch_1_option}"]["channels"]
        teacher_1_model = DenseNetTeacher(num_classes, channels)
    elif tch_1_option == "r":
        dim = params["model"][f"tch_{tch_1_option}"]["dim"]
        channels = params["model"][f"tch_{tch_1_option}"]["channels"]
        teacher_1_model = resnet50(num_classes, channels)
    elif tch_1_option == "v":
        dim = params["model"][f"tch_{tch_1_option}"]["dim"]
        depth = params["model"][f"tch_{tch_1_option}"]["depth"]
        heads = params["model"][f"tch_{tch_1_option}"]["heads"]
        mlp_dim = params["model"][f"tch_{tch_1_option}"]["mlp-dim"]
        channels = params["model"][f"tch_{tch_1_option}"]["channels"]
        teacher_1_model = TMAP(
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
    #     teacher_model = tch_m

    if tch_2_option == "d":
        channels = params["model"][f"tch_{tch_2_option}"]["channels"]
        teacher_2_model = DenseNetTeacher(num_classes, channels)
    elif tch_2_option == "r":
        dim = params["model"][f"tch_{tch_2_option}"]["dim"]
        channels = params["model"][f"tch_{tch_2_option}"]["channels"]
        teacher_2_model = resnet50(num_classes, channels)
    elif tch_2_option == "v":
        dim = params["model"][f"tch_{tch_2_option}"]["dim"]
        depth = params["model"][f"tch_{tch_2_option}"]["depth"]
        heads = params["model"][f"tch_{tch_2_option}"]["heads"]
        mlp_dim = params["model"][f"tch_{tch_2_option}"]["mlp-dim"]
        channels = params["model"][f"tch_{tch_2_option}"]["channels"]
        teacher_2_model = TMAP(
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
    #     teacher_model = tch_m

    if tch_3_option == "d":
        channels = params["model"][f"tch_{tch_3_option}"]["channels"]
        teacher_3_model = DenseNetTeacher(num_classes, channels)
    elif tch_3_option == "r":
        dim = params["model"][f"tch_{tch_3_option}"]["dim"]
        channels = params["model"][f"tch_{tch_3_option}"]["channels"]
        teacher_3_model = resnet50(num_classes, channels)
    elif tch_3_option == "v":
        dim = params["model"][f"tch_{tch_3_option}"]["dim"]
        depth = params["model"][f"tch_{tch_3_option}"]["depth"]
        heads = params["model"][f"tch_{tch_3_option}"]["heads"]
        mlp_dim = params["model"][f"tch_{tch_3_option}"]["mlp-dim"]
        channels = params["model"][f"tch_{tch_3_option}"]["channels"]
        teacher_3_model = TMAP(
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
    #     teacher_model = tch_m

    if tch_4_option == "d":
        channels = params["model"][f"tch_{tch_4_option}"]["channels"]
        teacher_4_model = DenseNetTeacher(num_classes, channels)
    elif tch_4_option == "r":
        dim = params["model"][f"tch_{tch_4_option}"]["dim"]
        channels = params["model"][f"tch_{tch_4_option}"]["channels"]
        teacher_4_model = resnet50(num_classes, channels)
    elif tch_4_option == "v":
        dim = params["model"][f"tch_{tch_4_option}"]["dim"]
        depth = params["model"][f"tch_{tch_4_option}"]["depth"]
        heads = params["model"][f"tch_{tch_4_option}"]["heads"]
        mlp_dim = params["model"][f"tch_{tch_4_option}"]["mlp-dim"]
        channels = params["model"][f"tch_{tch_4_option}"]["channels"]
        teacher_4_model = TMAP(
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
    #     teacher_model = tch_m

    if stu_option == "d":
        channels = params["model"][f"stu_{stu_option}"]["channels"]
        model = DenseNetStudent(num_classes, channels)
    elif stu_option == "r":
        dim = params["model"][f"stu_{stu_option}"]["dim"]
        channels = params["model"][f"stu_{stu_option}"]["channels"]
        model = resnet_tiny(num_classes, channels)
    elif stu_option == "v":
        dim = params["model"][f"stu_{stu_option}"]["dim"]
        depth = params["model"][f"stu_{stu_option}"]["depth"]
        heads = params["model"][f"stu_{stu_option}"]["heads"]
        mlp_dim = params["model"][f"stu_{stu_option}"]["mlp-dim"]
        channels = params["model"][f"stu_{stu_option}"]["channels"]
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

    tch_1_model_save_path = os.path.join(model_dir, "teacher_1.pth")
    tch_2_model_save_path = os.path.join(model_dir, "teacher_2.pth")
    tch_3_model_save_path = os.path.join(model_dir, "teacher_3.pth")
    tch_4_model_save_path = os.path.join(model_dir, "teacher_4.pth")
    
    model_save_path = os.path.join(model_dir, "student.pth")

    train_loader = torch.load(os.path.join(processed_dir, "train_loader_stu"))
    test_loader = torch.load(os.path.join(processed_dir, "test_loader_stu"))

    model = model.to(device)
    teacher_1_model = teacher_1_model.to(device)
    teacher_2_model = teacher_2_model.to(device)
    teacher_3_model = teacher_3_model.to(device)
    teacher_4_model = teacher_4_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loading = False

    with Live(dir="res", resume=True) as live:
        live.step = 1
        run_epoch(epochs, early_stop, loading, tch_1_model_save_path, tch_2_model_save_path, tch_3_model_save_path, tch_4_model_save_path, model_save_path, train_loader, test_loader, live)

if __name__ == "__main__":
    main()