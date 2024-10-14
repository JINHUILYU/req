import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel


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


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()


texts = [
    '''The system shall trigger an alert after the network connection is lost for more than 5 seconds "if" the network connection is not restored within 10 seconds.''',
    '''The readout Fuel_L_Tank_Qty_Rdt shall display the Text Value in the LB1958 CUI State "as" calculated using the logic in the table titled "Fuel_L_Tank_Qty_Rdt Logic".''']
model = torch.load(f"E:/Project/req/models/bert_classifier.pth",
                   map_location="cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = f'E:/Project/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for text in texts:
    sentiment = predict_sentiment(text, model, tokenizer, device)
    print(f"text: {text} label: {sentiment}")
