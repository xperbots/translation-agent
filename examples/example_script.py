import os
from dotenv import load_dotenv
import anthropic
from icecream import ic

if __name__ == "__main__":
    load_dotenv()  # read local .env file
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    source_lang, target_lang, country = "Chinese", "English", "United States"
    relative_path = "sample-texts/sample-long1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
    Do not provide any explanations or text apart from the translation.
    {source_lang}: {source_text}
    {target_lang}:"""

    # 用户消息
    user_message = {"role": "user", "content": translation_prompt}

    # 发送消息给 Claude，并获取响应
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[user_message],
            system=system_message  # 将系统信息作为顶级参数传递
        )
        print(response.content)  # 打印 Claude 的响应
    except Exception as e:
        print(f"Error: {str(e)}")  # 错误处理



'''
import os
import translation_agent as ta

if __name__ == "__main__":
    source_lang, target_lang, country = "Chinese", "Vietnamese", "Vietnam"

    relative_path = "sample-texts/sample-long1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    
    translation = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )
 
    
    translation = ta.translate(source_lang, target_lang, source_text,country)
    #translation = ta.new_utils.one_chunk_initial_translation_2(source_text)
    
    translation_file_path = os.path.join(script_dir, "Final_Translation.txt")
    with open(translation_file_path, "w", encoding="utf-8") as translation_file:
        translation_file.write(translation)
    
    
    print(f"Translation Complete\n\n")
'''