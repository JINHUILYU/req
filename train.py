import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


# 定义数据集类
def load_data(data_file):
    df = pd.read_csv(data_file)[:]  # 从训练数据文件中读取数据
    texts = df['Sentence'].tolist()
    labels = [1 if sentiment == 1 else 0 for sentiment in df['Label'].tolist()]
    # 生成打乱的索引
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    # 打乱数据，使用打乱的索引重排文本和标签
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels


data_file = "dataset/train.csv"  # 训练数据集
texts, labels = load_data(data_file)
print(f"Number of samples: {len(texts)}")


# 自定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


# 自定义BERT分类器
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# 定义训练函数
def train(model, data_loader, optimizer, scheduler, device):
    model.train()  # 设置模型为训练模式
    for batch in data_loader:
        optimizer.zero_grad()  # 梯度清零
        input_ids = batch['input_ids'].to(device)  # input_ids是输入文本的编码, 有batch_size个文本，每个文本的长度为max_length
        attention_mask = batch['attention_mask'].to(device)  # attention_mask是输入文本的mask
        labels = batch['label'].to(device)  # labels是输入文本的标签
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # 模型输出
        loss = nn.CrossEntropyLoss()(outputs, labels)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率


# 定义评估函数
def evaluate(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # 模型输出
            _, preds = torch.max(outputs, dim=1)  # 求最大值、最大值的索引
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# 定义模型参数
bert_model_name = f'E:/Project/bert-base-uncased'
batch_size = 8  # 每次处理的样本数量
max_length = 128  # 每个样本的维度，少于128，则填0
# 所以每个输入文本的维度是：[batch_size, max_length]
num_classes = 2  # 分类数
num_epochs = 10  # 训练轮数作用：1. 控制训练时间 2. 控制模型性能
learning_rate = 2e-5  # 学习率作用：1. 控制模型参数更新的速度 2. 控制模型性能
# 加载和切分数据
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
# 初始化分词器，数据集，数据加载器
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
# 设置设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
# 设置优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
best_acc = 0.0
# 训练模型
print("Start training models.")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model, f"E:/Project/req/models/best_bert_classifier.pth")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
# 保存模型
torch.save(model, f"E:/Project/req/models/bert_classifier.pth")
