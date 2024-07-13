import os
import translation_agent as ta
from flask import Flask, request, jsonify,render_template
from icecream import ic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    source_text = data['original_text']
    source_lang = "Chinese"
    target_lang = data['target_language']
    country = data['target_country']
    llm_model= data['model']
    
    #ic(target_lang)
    #ic(country)
    #ic(llm_model)
    

    # Perform the translation using some translation library or API
    translation_result = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
        llm_model=llm_model
    )

    #完成一篇文章日志显示翻译完成   
    ic("翻译完成")
    
    return jsonify({'translated_text': translation_result})

if __name__ == '__main__':
    app.run(debug=True)


