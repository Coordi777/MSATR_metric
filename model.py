import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# 定义一个卷积神经网络模块，用于特征提取
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, 256)

    def forward(self, x):
        # 输入x的形状为[batch_size, 1, 128, 128]
        x = self.conv1(x)  # 输出形状为[batch_size, 16, 128, 128]
        x = F.relu(x)
        x = self.pool1(x)  # 输出形状为[batch_size, 16, 64, 64]
        x = self.conv2(x)  # 输出形状为[batch_size, 32, 64, 64]
        x = F.relu(x)
        x = self.pool2(x)  # 输出形状为[batch_size, 32, 32, 32]
        x = self.conv3(x)  # 输出形状为[batch_size, 64, 32, 32]
        x = F.relu(x)
        x = self.pool3(x)  # 输出形状为[batch_size, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc(x)  # 全连接层，输出形状为[batch_size, 256]
        return x


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        # 输入x的形状为[batch_size, 1, 128, 128]
        x = self.conv1(x)  # 输出形状为[batch_size, 32, 128, 128]
        x = F.relu(x)
        x = self.pool(x)  # 输出形状为[batch_size, 32, 64, 64]
        x = self.conv2(x)  # 输出形状为[batch_size, 64, 64, 64]
        x = F.relu(x)
        x = self.pool(x)  # 输出形状为[batch_size, 64, 32, 32]
        x = self.conv3(x)  # 输出形状为[batch_size, 128, 32, 32]
        x = F.relu(x)
        x = self.pool(x)  # 输出形状为[batch_size, 128, 16, 16]
        x = self.conv4(x)  # 输出形状为[batch_size, 256, 16, 16]
        x = F.relu(x)
        x = self.pool(x)  # 输出形状为[batch_size, 256, 8, 8]
        x = self.conv5(x)  # 输出形状为[batch_size, 512, 8, 8]
        x = F.relu(x)
        x = x.view(-1, 512 * 8 * 8)
        x = self.fc1(x)  # 全连接层，输出形状为[batch_size, 1024]
        x = F.relu(x)
        x = self.fc2(x)  # 全连接层，输出形状为[batch_size, 512]
        x = F.relu(x)
        x = self.fc3(x)  # 全连接层，输出形状为[batch_size, 1]
        return x


# 定义一个分类分支模块，用于对拼接后的特征向量进行分类
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 全连接层，输入维度为512（两个特征向量拼接），输出维度为10（假设有10个类别）
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # 输入x的形状为[batch_size, 512]
        x = self.fc(x)  # 全连接层，输出形状为[batch_size, 10]
        x = F.softmax(x, dim=1)  # softmax函数，输出形状为[batch_size, 10]
        return x


# 定义一个角度检测分支模块，用于对拼接后的特征向量进行回归
class AngleDetector(nn.Module):
    def __init__(self):
        super(AngleDetector, self).__init__()
        # 全连接层，输入维度为512（两个特征向量拼接），输出维度为1（角度差）
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # 输入x的形状为[batch_size, 512]
        x = self.fc(x)  # 全连接层，输出形状为[batch_size, 1]
        return x


# 定义一个协作模块，用于将分类分支和角度检测分支的输出进行信息交互
class Collaborator(nn.Module):
    def __init__(self):
        super(Collaborator, self).__init__()
        # 全连接层，输入维度为11（分类分支的输出和角度检测分支的输出拼接），输出维度为10（分类分支的新输出）
        self.fc1 = nn.Linear(55, 10)
        # 全连接层，输入维度为11（分类分支的输出和角度检测分支的输出拼接），输出维度为1（角度检测分支的新输出）
        self.fc2 = nn.Linear(55, 1)

    def forward(self, x1, x2):
        # 输入x1的形状为[batch_size, 10]，是分类分支的输出
        # 输入x2的形状为[batch_size, 1]，是角度检测分支的输出
        x = torch.cat((x1, x2), dim=1)  # 拼接操作，输出形状为[batch_size, 11]
        x1 = self.fc1(x)  # 全连接层，输出形状为[batch_size, 10]
        x2 = self.fc2(x)  # 全连接层，输出形状为[batch_size, 1]
        return x1, x2


# 定义一个双输入卷积神经网络模型，用于同时进行分类和角度检测
class DualInputCNN(nn.Module):
    def __init__(self):
        super(DualInputCNN, self).__init__()
        # 特征提取模块，用于对两张图片进行特征提取
        self.dcnn = DeepCNN()
        self.cnn = CNN()
        # 分类分支模块，用于对拼接后的特征向量进行分类
        self.classifier = Classifier()
        # 角度检测分支模块，用于对拼接后的特征向量进行回归
        self.angle_detector = AngleDetector()
        # 协作模块，用于将分类分支和角度检测分支的输出进行信息交互
        self.collaborator = Collaborator()

    def forward(self, x1, x2):
        # 输入x1和x2的形状都是[batch_size, 1, 128, 128]，是两张图片
        x1 = self.dcnn(x1)  # 特征提取模块，输出形状为[batch_size, 256]
        x2 = self.dcnn(x2)  # 特征提取模块，输出形状为[batch_size, 256]
        x = torch.cat((x1, x2), dim=1)  # 拼接操作，输出形状为[batch_size, 512]
        y1 = self.classifier(x)  # 分类分支模块，输出形
        y2 = self.angle_detector(x)  # 角度检测分支模块，输出形状为[batch_size, 1]
        y1, y2 = self.collaborator(y1, y2)  # 协作模块，输出形状为[batch_size, 10]和[batch_size, 1]
        # y1 = F.softmax(y1, dim=1)  # softmax函数，输出形状为[batch_size, 10]
        y2 = 45 * torch.sigmoid(y2)
        return y1, y2


# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 定义三个线性层，用于计算查询、键和值
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        # 定义一个缩放因子，用于归一化注意力分数
        self.scale = torch.sqrt(torch.tensor(in_channels // 8, dtype=torch.float))
        # 定义一个线性层，用于输出最终的特征图
        self.output = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 前向传播过程
        batch_size, channels, height, width = x.size()  # 获取输入的尺寸
        # 计算查询、键和值
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # 将二维特征图展平为一维向量，并转置
        key = self.key(x).view(batch_size, -1, height * width)  # 将二维特征图展平为一维向量
        value = self.value(x).view(batch_size, -1, height * width)  # 将二维特征图展平为一维向量
        # 计算注意力分数
        attention = torch.bmm(query, key) / self.scale  # 使用矩阵乘法计算查询和键的点积，并除以缩放因子
        attention = torch.softmax(attention, dim=-1)  # 对最后一个维度进行softmax操作，得到归一化的注意力分数
        # 计算加权的值
        weighted_value = torch.bmm(value, attention.permute(0, 2, 1))  # 使用矩阵乘法计算值和注意力分数的乘积，并转置
        weighted_value = weighted_value.view(batch_size, channels, height, width)  # 将一维向量还原为二维特征图
        # 计算输出的特征图
        output = self.output(weighted_value) + x  # 使用线性层对加权的值进行变换，并加上残差连接
        return output


# 定义模型类
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # 定义基础模型，使用预训练的ResNet18
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 冻结基础模型的参数，不参与训练
        for param in self.base_model.parameters():
            param.requires_grad = False
        # 获取基础模型的输出通道数
        out_channels = self.base_model.fc.in_features
        # 替换基础模型的全连接层，输出类别预测和角度预测
        self.base_model.fc = nn.Linear(out_channels, num_classes + 10)
        # 定义自注意力层，输入通道数为基础模型的输出通道数
        self.attention = SelfAttention(out_channels)
        # 定义全连接层，输入维度为基础模型的输出通道数乘以特征图的大小，输出维度为类别数加上角度数
        self.fc = nn.Linear(128 * 8 * 8, num_classes + 10)

    def forward(self, x):
        # 前向传播过程
        x = self.base_model.conv1(x)  # 使用基础模型的第一个卷积层
        x = self.base_model.bn1(x)  # 使用基础模型的第一个批归一化层
        x = self.base_model.relu(x)  # 使用基础模型的激活函数
        x = self.base_model.maxpool(x)  # 使用基础模型的最大池化层
        x = self.base_model.layer1(x)  # 使用基础模型的第一个残差块
        x = self.base_model.layer2(x)  # 使用基础模型的第二个残差块
        x = self.base_model.layer3(x)  # 使用基础模型的第三个残差块
        x = self.base_model.layer4(x)  # 使用基础模型的第四个残差块
        x = self.attention(x)  # 添加自注意力层
        x = x.view(-1, 128 * 8 * 8)  # 将二维特征图展平为一维向量
        x = self.fc(x)  # 使用全连接层输出类别预测和角度预测
        return x
