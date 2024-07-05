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

    # 将翻译结果输出到Translation1.txt文件中
    translation_file_path = os.path.join(script_dir, "Translation1.txt")
    with open(translation_file_path, "w", encoding="utf-8") as translation_file:
        translation_file.write(translation)
    print(f"Translation Complete\n\n")