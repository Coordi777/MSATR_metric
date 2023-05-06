import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from model import DualInputCNN,ResNet
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

BestEpoch = 0
BestSim = 0
classification_loss = nn.CrossEntropyLoss()
angle_detection_loss = nn.MSELoss()
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])


class CsvDataset(Dataset):
    def __init__(self, csv_file, transform=transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path = self.data.iloc[idx, 0]
        img2_path = self.data.iloc[idx, 1]
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        angle = self.data.iloc[idx, 2]

        label = self.data.iloc[idx, 3]
        # 将label和angle转化为tensor
        label = torch.tensor(label, dtype=torch.long)
        angle = torch.tensor(angle, dtype=torch.float32)

        return img1, img2, angle, label


def criterion(y_pred, y_true, log_vars):
    loss = torch.tensor(0, dtype=torch.float32, device=y_pred[0].device)
    for i in range(len(y_pred)):
        precision = torch.exp(-log_vars[i])
        diff = classification_loss(y_pred[i], y_true[i])
        loss += torch.sum(precision * diff + log_vars[i], -1)

    return torch.mean(loss)


# model = DualInputCNN().to('cuda:0')
model = ResNet(10).to('cuda:0')
model.train()
optimizer = AdamW(model.parameters(), lr=0.001)
all_dataset = CsvDataset('samples17_clean_angle10.csv')
train_size = int(0.8 * len(all_dataset))
test_size = len(all_dataset) - train_size
train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])
trian_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
EPOCH = 50
nums_triansteps = len(trian_dataloader) * EPOCH
pross_bar = tqdm(total=nums_triansteps)
step = 0
lr_scheduler = StepLR(optimizer, step_size=len(trian_dataloader) * 5, gamma=0.9)
min_loss = 10000
optimizer.zero_grad()
for epoch in range(EPOCH):
    for i in trian_dataloader:
        img1, img2, angle, label = i[0].to('cuda:0'), i[1].to('cuda:0'), i[2].to('cuda:0'), i[3].to('cuda:0')
        img = torch.cat((img1, img2), dim=1)
        # out = model(img1, img2)
        # label_pre, angel_pre = out[0], out[1]
        outputs = model(img)
        class_outputs = outputs[:, :-1]  # 最后一列是角度预测，其余是类别预测
        angle_outputs = outputs[:, -1]  # 最后一列是角度预测

        loss_cls = classification_loss(class_outputs, label)
        loss_ang = angle_detection_loss(angle_outputs.squeeze(), angle)
        # weight_an = loss_cls.item() / (loss_ang.item() + loss_cls.item())
        # weight_cls = loss_ang.item() / (loss_ang.item() + loss_cls.item())
        # loss = weight_cls * loss_cls + weight_an * loss_ang
        loss = loss_cls + loss_ang
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        lr_scheduler.step()
        pross_bar.update(1)
        pross_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0], loss_l=loss_cls.item(),
                              loss_an=loss_ang.item())
    # Test the model
    test_loss = 0.0
    test_acc_label = 0.0
    test_angel_loss = 0.0
    test_label_loss = 0.0
    with torch.no_grad():
        model.eval()
        for i in test_dataloader:
            img1, img2, angle, label = i[0].to('cuda:0'), i[1].to('cuda:0'), i[2].to('cuda:0'), i[3].to('cuda:0')
            # out = model(img1, img2)
            # label_pre, angel_pre = out[0], out[1]
            img = torch.cat((img1, img2), dim=1)
            outputs = model(img)
            class_outputs = outputs[:, :-1]  # 最后一列是角度预测，其余是类别预测
            angle_outputs = outputs[:, -1]  # 最后一列是角度预测

            loss_cls_e = classification_loss(class_outputs, label)
            loss_ang_e = angle_detection_loss(angle_outputs.squeeze(), angle)
            test_angel_loss += loss_ang_e.item()
            test_label_loss += loss_cls_e.item()
            loss_e = loss_cls_e + loss_ang_e

            label_pre = F.softmax(class_outputs, dim=1)

            test_loss += loss_e.item()
            # _, predicted_label = torch.max(out[0], 1)
            _, predicted_label = torch.max(class_outputs, 1)
            test_acc_label += (predicted_label == label).sum().item()

        test_loss /= len(test_dataloader)
        test_angel_loss /= len(test_dataloader)
        test_label_loss /= len(test_dataloader)
    if test_loss < min_loss:
        min_loss = test_loss
        BestEpoch = epoch + 1
    if epoch - 3 > BestEpoch:
        print(f"early stop at {epoch + 1} with best epoch {BestEpoch} and test similarity {min_loss}.")
        break

    test_acc_label /= len(test_dataset)
    model.train()
    print('Test loss: %.3f' % test_loss)
    print('Test accuracy label: %.3f' % test_acc_label)
    print('Test angel loss: %.3f' % test_angel_loss)
    print('Test label loss: %.3f' % test_label_loss)
