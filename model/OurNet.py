import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 使用PyG的GCN层
from mamba_ssm import Mamba, Mamba2
from transformers import AutoModel


# def TextEncoder():
#     model = AutoModel.from_pretrained("./MedBERT", output_hidden_states=False)
#     return model

# def ImageEncoder(d_model):
#     return ResNet3D(Bottleneck3D, [2, 2, 3, 3], num_input_channels=1, num_features=d_model)


class TabularEncoder(nn.Module):
    """表格数据特征提取器"""
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
class ImageEncoder(nn.Module):
    """3D卷积特征提取器"""
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class TemporalFusion(nn.Module):
    def __init__(self, feature_size):
        super(TemporalFusion, self).__init__()
        
        # t1 特征处理: 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # 3D卷积
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=feature_size, kernel_size=2, stride=2)
        
        # 残差层：继续使用卷积代替Linear
        self.res_conv1 = nn.Conv1d(in_channels=feature_size, out_channels=1, kernel_size=3, padding=1)
        self.res_conv2 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        
        # 可学习权重
        self.weight_t0 = nn.Parameter(torch.tensor(0.7))  # t0 的权重
        self.weight_t1 = nn.Parameter(torch.tensor(0.3))  # t1 的权重
        
        self.relu = nn.ReLU()

    def alternate_concat(self, tensor1, tensor2):
        batch_size, l, dim = tensor1.size()
        # assert dim == 512, "Both tensors must have a dimension of 512"
        assert tensor1.size() == tensor2.size(), "Both tensors must have the same shape"
    
        # Create an empty tensor to store the result with double the dimension
        result = torch.zeros((batch_size, l, dim * 2), dtype=tensor1.dtype, device=tensor1.device)
    
        # # Alternate elements from tensor1 and tensor2
        # result[:,:, 0::2] = tensor1  # Even indices come from tensor1
        # result[:,:, 1::2] = tensor2  # Odd indices come from tensor2
        result = torch.concat([tensor1, tensor2], dim=2)  # Concatenate along the last dimension
        return result

    def forward(self, t0, t1):
        # t1 特征的处理: 上采样 -> 平均池化 -> 3D卷积
        # t0 = x[:,0:1]
        # t1 = x[:,1:2]
        # fush = torch.cat([t0, t1], dim=1)
        # print(t0.shape)
        fused = self.alternate_concat(t0, t1)  # t1 特征的交替拼接
        # print(t1.shape ,fused.shape)
        fused = self.relu(self.conv1d(fused))  # t1 1D卷积
        # print(fused.shape)
        fused = self.res_conv1(fused)  # 残差连接
        # print(fused.shape)

        # 第一层残差连接（保留3D特征形状，使用卷积）
        #t1_fused = self.relu(t1_conv + self.res_conv1(t1_conv))  # 残差连接

        # 可学习加权融合（保持3D形状进行加权融合）
        # weight_t0 = torch.sigmoid(self.weight_t0)
        # weight_t1 = torch.sigmoid(self.weight_t1)
        weights = torch.softmax(torch.stack([self.weight_t0, self.weight_t1]), dim=0)  # 在第0维上进行softmax
        weight_t0, weight_t1 = weights[0], weights[1]

        fused_feature = weight_t0 * fused + weight_t1 * t1  # 加权融合

        return fused_feature.squeeze(1)

class MultiModalFusion(nn.Module):
    """多模态融合模块+Mamba"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.norm = nn.LayerNorm(d_model)
        self.fuse = TemporalFusion(d_model)
        
    def forward(self, img_feat, tab_feat):
        # 将两个特征作为序列输入

        # x = torch.stack([img_feat, tab_feat], dim=1)  # (batch, 2, d_model)
        # print(x.shape)
        # x = self.mamba(x)
        
        # attn_output, _ = self.attn(x, x, x)
        fused = self.fuse(img_feat, tab_feat)
        # print(fused.shape)
        return fused

class GraphConstructor(nn.Module):
    """自适应图构造模块"""
    def __init__(self, in_dim, th=0.95):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.th = th

    def forward(self, features):
        # 特征投影和归一化
        features = F.normalize(self.proj(features), dim=1)
        # print("features shape:", features.shape)
        # 计算余弦相似度矩阵
        adj = torch.mm(features, features.t())
        # print(adj)
        # 应用阈值生成二值图
        adj = (adj > self.th).float()
        # 去除自连接
        adj = adj - torch.diag(torch.diag(adj))
        return adj

class GraphConvBlock(nn.Module):
    """图卷积模块"""
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
class OurNet(nn.Module):
    """完整模型架构"""
    def __init__(self, img_dim, tab_dim, d_model=768, nhead=4):
        super().__init__()
        # 特征提取器
        self.image_encoder = ImageEncoder(d_model)
        self.tabular_encoder = TabularEncoder(tab_dim, d_model)
        
        self.image_mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.tab_mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )  
        # 多模态融合
        self.fusion = MultiModalFusion(d_model, nhead)
        
        # 图构造
        self.graph_constructor = GraphConstructor(d_model)
        
        # 图卷积
        self.graph_conv = GraphConvBlock(d_model, d_model//2, d_model)
        
        # 分类头
        self.classifier = nn.Linear(d_model, 2)
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 小常数避免数值问题
        self.eps = 1e-8

    def compute_csdm_loss(self, img_feat, tab_feat):
        """
        计算双向CSDM损失 (text-to-vision和vision-to-text)
        
        参数:
            img_feat: 图像特征 (batch_size, d_model)
            tab_feat: 表格特征 (batch_size, d_model)
            
        返回:
            csdm_loss: 双向CSDM损失
        """
        # 归一化特征
        img_feat_norm = F.normalize(img_feat, p=2, dim=1)
        tab_feat_norm = F.normalize(tab_feat, p=2, dim=1)
        
        # 计算相似度矩阵 (batch_size, batch_size)
        sim_matrix = torch.matmul(tab_feat_norm, img_feat_norm.t()) / self.temperature
        
        # text-to-vision概率分布
        t2v_p = F.softmax(sim_matrix, dim=1)  # 每行是text对images的分布
        t2v_q = torch.eye(sim_matrix.size(0), device=sim_matrix.device)  # 真实匹配是对角矩阵
        
        # vision-to-text概率分布
        v2t_p = F.softmax(sim_matrix.t(), dim=1)  # 每行是image对texts的分布
        v2t_q = torch.eye(sim_matrix.size(0), device=sim_matrix.device)  # 真实匹配是对角矩阵
        
        # 计算KL散度损失
        t2v_loss = F.kl_div((t2v_p + self.eps).log(), t2v_q, reduction='mean')
        v2t_loss = F.kl_div((v2t_p + self.eps).log(), v2t_q, reduction='mean')
        
        return t2v_loss + v2t_loss

    def forward(self, image, text):
        # 特征提取
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(text)
        # print(img_feat.shape, tab_feat.shape)
        img_feat = self.image_mamba(img_feat.unsqueeze(1))  # (batch, 1, d_model)
        tab_feat = self.tab_mamba(tab_feat.unsqueeze(1))  # (batch, 1, d_model)
        # print(img_feat.shape, tab_feat.shape)
        # 计算CSDM损失
        csdm_loss = self.compute_csdm_loss(img_feat.squeeze(1), tab_feat.squeeze(1))
        
        # 多模态融合
        fused = self.fusion(img_feat, tab_feat)
        
        # 构建图
        adj_matrix = self.graph_constructor(fused)
        edge_index = adj_matrix.nonzero().t()
        
        # 图卷积处理
        node_embeddings = self.graph_conv(fused, edge_index)
        
        # 分类输出
        logits = self.classifier(node_embeddings)
        logits = F.softmax(logits, dim=1)
        return adj_matrix, node_embeddings, logits, csdm_loss
        # return {
        #     'adj_matrix': adj_matrix,
        #     'node_embeddings': node_embeddings,
        #     'logits': logits,
        #     'csdm_loss': csdm_loss
        # }


from torchvision.models.resnet import Bottleneck

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_input_channels=1, zero_init_residual=False, num_features=512):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        
        # 修改第一个卷积层为3D卷积，处理体积数据
        self.conv1 = nn.Conv3d(num_input_channels, 64, kernel_size=(3, 7, 7), 
                              stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # 3D版本的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 自适应平均池化将空间和时间维度降为1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 全连接层输出512维特征
        self.fc = nn.Linear(512 * block.expansion, num_features)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 零初始化每个残差分支中的最后一个BN
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状: (batch_size, 1, 20, 256, 256)
        x = self.conv1(x)    # (batch_size, 64, 20, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, 64, 20, 64, 64)
        
        x = self.layer1(x)   # (batch_size, 256, 20, 64, 64)
        x = self.layer2(x)   # (batch_size, 512, 20, 32, 32)
        x = self.layer3(x)   # (batch_size, 1024, 20, 16, 16)
        x = self.layer4(x)    # (batch_size, 2048, 20, 8, 8)
        
        # 全局平均池化
        x = self.avgpool(x)  # (batch_size, 2048, 1, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 2048)
        
        # 全连接层
        x = self.fc(x)       # (batch_size, 512)
        
        return x

# 3D版本的Bottleneck块
class Bottleneck3D(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__(inplanes, planes, stride, downsample)
        
        # 将所有2D卷积替换为3D卷积
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)