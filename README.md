### 简介

这是一个基于kaggle上的[“Daily News for Stock Market Prediction”](https://www.kaggle.com/datasets/aaron7sun/stocknews)数据集的工作。

由于传统方法（如LSTM+历史股价）对于股市的预测存在滞后性显著（依赖滞后1日以上的价格数据）和噪声敏感度高（易受高频交易噪音干扰）的缺陷。所以我们希望通过基于NLP的新闻预测则通过实时捕捉突发事件（如政策调整、行业危机）和量化市场情绪（BERT情感分析）实现前瞻性建模。

我们从学习的角度，提出了从完全的基础建模方式->到进阶的机器学习（IF—IDF+SVM）模式->以及基于BERT预训练模型的深度学习模式。

这个代码库是基于BERT预训练模型的深度学习模式的总结和demo。

[我们的notebook](https://www.kaggle.com/code/kj294443/the-project-of-west-ice-storage/notebook)

### 使用demo
运行以下命令
```shell
git clone https://github.com/ghostdoglzd/The_project_of_West_Ice_Storage.git
cd The_project_of_West_Ice_Storage
python app.py
```
然后打开`http://localhost:5000`，即可使用demo。

