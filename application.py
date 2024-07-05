import os
import translation_agent as ta
from flask import Flask, request, jsonify,render_template


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
    
    '''
    relative_path = "examples/sample-texts/sample-long1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)
    
    with open(full_path, encoding="utf-8") as file:s
        source_text = file.read()
    '''
    # Perform the translation using some translation library or API
    translation_result = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )

    '''
    # 将翻译结果输出到Translation1.txt文件中
    translation_file_path = os.path.join(script_dir, "Translation1.txt")
    with open(translation_file_path, "w", encoding="utf-8") as translation_file:
        translation_file.write(translation)
    '''
    
    print(f"Translation Complete\n\n")
    
    return jsonify({'translated_text': translation_result})

if __name__ == '__main__':
    app.run(debug=True)


