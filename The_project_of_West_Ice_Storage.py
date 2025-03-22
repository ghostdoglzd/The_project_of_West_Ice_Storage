import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import re
from textblob import TextBlob

# ���ݼ���
data = pd.read_csv(r'D:\The_project_of_West_Ice_Storage\Combined_News_DJIA.csv')


# ��ʼ��BERT�ִ���
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# �ı���ϴ����
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?$%-]+', ' ', text)
    return text.strip()

# ��ȡTextBlob����
def extract_textblob_features(text):
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity,
        'word_count': len(text.split()),
        'exclamation': text.count('!'),
        'question': text.count('?'),
        'dollar_sign': text.count('$')
    }

# Ԥ��������
print("Processing text features...")
textblob_features = []
for col in data.filter(regex=r"^Top").columns:
    data[col] = data[col].apply(lambda x: clean_text(str(x)))
    features = data[col].apply(lambda x: extract_textblob_features(str(x)))
    textblob_features.append(pd.DataFrame(features.tolist()))

# �ϲ�ÿ������
daily_features = []
for i in range(len(data)):
    day_features = {
        'polarity': np.mean([df.iloc[i]['polarity'] for df in textblob_features]),
        'subjectivity': np.mean([df.iloc[i]['subjectivity'] for df in textblob_features]),
        'word_count': np.sum([df.iloc[i]['word_count'] for df in textblob_features]),
        'exclamation': np.sum([df.iloc[i]['exclamation'] for df in textblob_features]),
        'question': np.sum([df.iloc[i]['question'] for df in textblob_features]),
        'dollar_sign': np.sum([df.iloc[i]['dollar_sign'] for df in textblob_features])
    }
    daily_features.append(day_features)

textblob_df = pd.DataFrame(daily_features)
feature_columns = textblob_df.columns.tolist()

# ��׼������
scaler = StandardScaler()
textblob_df[feature_columns] = scaler.fit_transform(textblob_df[feature_columns])

# �ϲ������ı�
print("Combining news texts...")
data["combined_news"] = data.filter(regex=r"^Top").apply(
    lambda x: f' {tokenizer.sep_token} '.join(
        str(content) for content in x if pd.notnull(content)
    ),
    axis=1
)

# ���ڴ���
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

# ���ݼ�����
print("Splitting datasets...")
train = data[data['Date'] <= '2014-12-31']
val = data[(data['Date'] > '2014-12-31') & (data['Date'] <= '2015-12-31')]
test = data[data['Date'] > '2015-12-31']

# ��ȡ��������
train_features = textblob_df.iloc[train.index].values
val_features = textblob_df.iloc[val.index].values
test_features = textblob_df.iloc[test.index].values

# BERT���뺯��
def bert_encode(texts, tokenizer, max_len=256):
    return tokenizer(
        texts.tolist(),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

# ���ݱ���
print("Encoding text data...")
train_encodings = bert_encode(train['combined_news'], tokenizer)
val_encodings = bert_encode(val['combined_news'], tokenizer)
test_encodings = bert_encode(test['combined_news'], tokenizer)

# ��ǩ����
train_labels = torch.tensor(train['Label'].values)
val_labels = torch.tensor(val['Label'].values)
test_labels = torch.tensor(test['Label'].values)

# �Զ������ݼ���
class HybridDataset(Dataset):
    def __init__(self, encodings, features, labels):
        self.encodings = encodings
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['text_features'] = self.features[idx]
        item['labels'] = self.labels[idx]
        return item

# ����DataLoader
print("Creating data loaders...")
train_dataset = HybridDataset(train_encodings, train_features, train_labels)
val_dataset = HybridDataset(val_encodings, val_features, val_labels)
test_dataset = HybridDataset(test_encodings, test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ���ģ�ͼܹ�
class HybridClassifier(nn.Module):
    def __init__(self, num_text_features, dropout=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-tone')
        
        # ����ǰ6��
        for layer in self.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # BERT�������
        self.bert_dropout = nn.Dropout(dropout)
        
        # �ı���������
        self.text_feature_processor = nn.Sequential(
            nn.Linear(num_text_features, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # ���Ϸ�����
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
        # ������ʼ��
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, input_ids, attention_mask, text_features):
        # BERT����
        outputs = self.bert(input_ids, attention_mask, return_dict=False)
        pooled_output = outputs[1]
        bert_output = self.bert_dropout(pooled_output)
        
        # �ı���������
        text_feature_output = self.text_feature_processor(text_features)
        
        # �����ں�
        combined = torch.cat((bert_output, text_feature_output), dim=1)
        return self.classifier(combined)

# �豸����
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ��ʧ����
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return loss.mean()

# ģ�ͳ�ʼ��
num_text_features = train_features.shape[1]
model = HybridClassifier(num_text_features=num_text_features).to(device)

# ѵ������
gradient_accumulation_steps = 4
epochs = 10
total_steps = len(train_loader) * epochs // gradient_accumulation_steps

# �Ż�������
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.05 * total_steps),
    num_training_steps=total_steps
)
loss_fn = FocalLoss(alpha=0.5)

# ѵ������
def train(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        text_features = batch['text_features'].to(device)
        labels = batch['labels'].to(device).float().unsqueeze(1)
        
        outputs = model(input_ids, attention_mask, text_features)
        loss = loss_fn(outputs, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(data_loader)-1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        total_loss += loss.item() * gradient_accumulation_steps
        
    return total_loss / len(data_loader)

# ��������
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_features = batch['text_features'].to(device)
            labels = batch['labels'].to(device).float().unsqueeze(1)
            
            outputs = model(input_ids, attention_mask, text_features)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            preds.extend((probs > 0.5).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    preds_np = np.array(preds).flatten()
    true_labels_np = np.array(true_labels).flatten()
    probs_np = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(true_labels_np, preds_np)
    f1 = f1_score(true_labels_np, preds_np)
    
    return total_loss / len(data_loader), accuracy, f1, probs_np, true_labels_np

# ���ӻ�����
def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(len(train_losses)), range(1, len(train_losses)+1))
    plt.tight_layout()
    plt.show()

# ѵ��ѭ��
print("\nStarting training...")
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(epochs):
    # ѵ���׶�
    train_loss = train(model, train_loader, optimizer, scheduler)
    
    # ��֤�׶�
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)
    
    # ��¼��ʧ
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # ��ͣ����
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # ��ӡ����
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f}")

# ��ʧ����
plot_loss_curve(train_losses, val_losses)

# ���ղ���
print("\nEvaluating on test set...")
test_loss, test_acc, test_f1, test_probs, test_true = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f}")

# ROC����
plot_roc_curve(test_true, test_probs)

# ģ�ͱ���
output_dir = 'D:\\py\\The_project_of_West_Ice_Storage\\model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
torch.save(model.state_dict(), os.path.join(output_dir, 'hybrid_model.pth'))
print(f"Model saved to {os.path.join(output_dir, 'hybrid_model.pth')}")
