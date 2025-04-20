# app.py
from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv
from xbk import Pre_model as Pre
from xbk.Generate_model import generate_report

load_dotenv()  # 加载环境变量

app = Flask(__name__)
DEEPSEEK_API_KEY = "sk-6e03af0b762a43ddb92aae83076907f8"

@app.route('/')
def home():
    return render_template('index.html')

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_deepseek_analysis(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": f"作为专业市场分析师，请对以下分析结果进行深度解读:\n{prompt}"}
        ],
        "temperature": 0.7
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=data,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return "API请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"API请求失败: {str(e)}"
    
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
        res = "今日大盘看涨" if prediction == 1 else "今日大盘看涨"
        report_data = generate_report(text)  # 修改为接收字典

        # 组合关键词标记的报告
        marked_report = report_data["report"]
        for keyword in report_data["keywords"]:
            marked_report = marked_report.replace(keyword, f"**{keyword}**")

        # 调用DeepSeek API
        analysis_prompt = f"基础分析结果: {res}\n详细报告: {marked_report}"
        deepseek_analysis = get_deepseek_analysis(analysis_prompt)

        return jsonify({
            'result': res,
            'report': marked_report,  # 包含**标记的报告
            'keywords': report_data["keywords"],  # 单独传递关键词列表
            'deepseek_analysis': deepseek_analysis
        })

    except Exception as e:
        return jsonify({'error': f"Runtime error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)