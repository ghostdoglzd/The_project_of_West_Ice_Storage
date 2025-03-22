from xbk.Pre_model import ModelWrapper
import os


# ʾ�������ı�
combined_news = """Apple announces new iPhone 16 with revolutionary AI features. NASA discovers signs of water on Mars, sparking new exploration plans. Global stock markets tumble as oil prices hit record highs. Elon Musk's Tesla unveils affordable electric pickup truck. Celebrity couple splits after 10 years of marriage, sources say. Major cyberattack hits government systems worldwide, causing chaos. Climate change summit in Paris ends with historic agreement. New study finds link between social media use and mental health decline. Olympic gold medalist tests positive for banned substance. Bitcoin price surges past $100,000 amid regulatory uncertainty. Controversial new law passed in European Union, facing backlash. Huge asteroid narrowly misses Earth, scientists relieved. Celebrated author releases long-awaited new novel. Major Hollywood studio announces all-female superhero film. Pandemic variant resurges, prompting new travel restrictions. Robotics company unveils humanoid assistant for home use. Political scandal rocks Capitol Hill, hearings begin. Record-breaking heatwave hits North America, breaking records. New wonder drug approved for treating Alzheimer's. SpaceX launches first all-civilian mission to the Moon. Major fast-food chain introduces plant-based burger. Fashion icon's latest collection goes viral on social media. Controversial video game banned in multiple countries. Scientists discover new species in Amazon rainforest. Global protests erupt over proposed internet censorship laws."""

def main():
    # ģ��·��
    model_path = "D:/The_project_of_West_Ice_Storage/model/model.pth"
    news_text = combined_news  # ʹ��ʾ�������ı�

    try:
        # ����ģ�ͷ�װ����
        model_wrapper = ModelWrapper(
            model_path,
            example_text=news_text  # ʹ�������ı���Ϊ��������ʾ��
        )

        # ����Ԥ��
        prediction = model_wrapper.predict(news_text)
        if prediction==1:
            res="rising"
        else:
            res="losing"
        print(f"Prediction result: {res}")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except RuntimeError as e:
        print(f"Model loading failed: {str(e)}")
    except Exception as e:
        print(f"Runtime error: {str(e)}")

if __name__ == "__main__":
    main()