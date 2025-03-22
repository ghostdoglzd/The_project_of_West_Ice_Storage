import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from textblob import TextBlob
from nltk.tokenize import sent_tokenize

class HybridClassifier(nn.Module):
    def __init__(self, num_text_features, dropout=0.2):
        super().__init__()
        # 加载完整的BERT模型（推理时不需要冻结层）
        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-tone')
        self.bert_dropout = nn.Dropout(dropout)
        
        # 文本特征处理模块
        self.text_feature_processor = nn.Sequential(
            nn.Linear(num_text_features, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 分类器（保持原有结构）
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

    def forward(self, input_ids, attention_mask, text_features):
        # 前向传播保持不变
        outputs = self.bert(input_ids, attention_mask, return_dict=False)
        pooled_output = outputs[1]
        bert_output = self.bert_dropout(pooled_output)
        text_feature_output = self.text_feature_processor(text_features)
        combined = torch.cat((bert_output, text_feature_output), dim=1)
        return self.classifier(combined)

class ModelWrapper:
    def __init__(self, model_path, example_text=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        
        # 自动检测特征维度
        demo_text = example_text or "Default text for feature dimension detection"
        self.num_text_features = len(self._extract_features(demo_text))
        
        # 初始化并加载模型
        self.model = HybridClassifier(self.num_text_features).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 设置为推理模式

    def _process_input(self, text):
        sentences = sent_tokenize(text)
        processed_text = f" {self.tokenizer.sep_token} ".join(sentences)
        return self.tokenizer(
            processed_text,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    def _extract_features(self, text):
        analysis = TextBlob(text)
        return [
            analysis.sentiment.polarity,       # 情感极性
            analysis.sentiment.subjectivity,   # 主观性
            len(text.split()),                 # 词数统计
            text.count('!'),                   # 感叹号计数
            text.count('?'),                   # 问号计数
            text.count('$')                    # 美元符号计数
        ]

    def predict(self, news_text, custom_features=None):
        # 文本编码
        inputs = self._process_input(news_text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 特征处理
        features = custom_features if custom_features else self._extract_features(news_text)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 执行推理
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features_tensor)
            probability = torch.sigmoid(output).item()
        
        return 1 if probability > 0.5 else 0