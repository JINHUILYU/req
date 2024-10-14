import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel


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


# 定义预测函数
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()


# 预测
# test_text = "The movie was great and I really enjoyed the performances of the actors."
model = torch.load(f"E:/Project/req/models/bert_classifier.pth", map_location="cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = f'E:/Project/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义数据集类
def load_data(data_file):
    df = pd.read_csv(data_file)[:]  # 从csv文件中读取数据, 读取1000-2000行
    texts = df['Sentence'].tolist()
    labels = [1 if sentiment == 1 else 0 for sentiment in df['Label'].tolist()]
    return texts, labels


data_file = "dataset/test.csv"
test_texts, test_labels = load_data(data_file)
count_correct = 0
for i in range(len(test_texts)):
    text = test_texts[i]
    label = test_labels[i]
    sentiment = predict_sentiment(text, model, tokenizer, device)
    print(f"text: {text}\tlabel: {label}\tpredict: {sentiment}")
    if label == sentiment:
        count_correct += 1
print('accuracy:', count_correct / len(test_texts))
