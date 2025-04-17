# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from xbk import Pre_model as Pre
from xbk.Generate_model import generate_report  # 新增导入
import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    text = data['text']
    try:
        # 原分类模型预测
        model_path = "./model/model.pth"
        model_wrapper = Pre.ModelWrapper(model_path, example_text=text)
        prediction = model_wrapper.predict(text)
        res = "今日大盘看涨" if prediction == 1 else "今日大盘看跌"

        # 新增生成报告
        report = generate_report(text)  # 调用生成报告函数

        return jsonify({
            'result': res,
            'report': report  # 添加报告内容
        })

    except FileNotFoundError:
        return jsonify({'error': f"Model file not found at {model_path}"}), 500
    except RuntimeError as e:
        return jsonify({'error': f"Model loading failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"Runtime error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)