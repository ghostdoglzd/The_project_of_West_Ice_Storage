# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from xbk import Pre_model as Pre
import torch
import nltk
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# model = Pre.HybridClassifier().to(device)
# state_dict = torch.load('model.pth', map_location=device)
# model.load_state_dict(state_dict)
# model.eval()

@app.route('/')
def home():
    return render_template('index.html')

#从前端拿到文本数据
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # 使用 get_json() 获取 JSON 数据
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    text = data['text']
    print(text)
    try:
        print("try")
        model_path = "./model/model.pth"#模型路径
        model_wrapper = Pre.ModelWrapper(
            model_path,
            example_text=text
        )
        print("try")
        prediction = model_wrapper.predict(text)
        res = "rising" if prediction == 1 else "losing"
        print(f"Prediction result: {res}")
        return jsonify({'result': res})

    except FileNotFoundError:
        return jsonify({'error': f"Model file not found at {model_path}"}), 500
    except RuntimeError as e:
        return jsonify({'error': f"Model loading failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"Runtime error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)