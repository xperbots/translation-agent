import os
from typing import List
from typing import Union
import time
import random
import difflib
import anthropic

import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter




load_dotenv()  # read local .env file
openaiclient = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claudclient = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

MAX_TOKENS_PER_CHUNK = (
    1900  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time

MAX_TOKENS_OVERALL = (
    60000  #
)
# discrete number of tokens to translate at a time




#default model for all kind of tasks in GPT completion
DEFAULT_MODEL = "gpt-4-turbo"

#model for splitting text into chunks
SPLIT_MODEL = "gpt-3.5-turbo"

#First translation model
FIRST_TRANSLATION_MODEL = "gpt-4-turbo"
#Second translation model
SECOND_TRANSLATION_MODEL_2 = "gpt-4-turbo"


#尝试使用gpt-3.5-turbo-instruct模型来翻译
def get_completion_instruct_model(prompt, model="gpt-3.5-turbo-instruct", max_tokens=2000, temperature=0):
    
    for i in range(3):  # 最多重试三次
        try:
            response = openaiclient.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        
        except openai.RateLimitError as e:
            ic(f"Rate limit exceeded (尝试 {i+1}/3): {str(e)}")
            if i < 2:
                time.sleep(5)  # 等待更长时间
        except openai.APIError as e:
            ic(f"OpenAI API错误 (尝试 {i+1}/3): {str(e)}")
            if i < 2:
                time.sleep(1)
        except openai.BadRequestError as e:
            ic(f"请求错误 (可能是 token 数量超限): {str(e)}")
            # 这里您可能需要调整 prompt 或 max_tokens
            return None
        except Exception as e:
            ic(f"未知错误: {str(e)}")
            return None
    
    ic("所有重试均失败")
    return None

def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    if json_mode:
        response = openaiclient.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = openaiclient.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def one_chunk_initial_gpt4_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion(prompt, system_message=system_message)

    return translation


def one_chunk_initial_gpt_instruct_translation(source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < MAX_TOKENS_PER_CHUNK:
        ic("Translating text as single chunk")

        prompt = f"""You are a professional translation engine. Please translate the text delimited by triple backticks into Vietnamese without explanation.
        Original content:```你好```
        Translated content:Xin chào
        Original content:```{source_text}```
        Translated content:"""
        
        final_translation = get_completion_instruct_model(prompt)

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=MAX_TOKENS_PER_CHUNK
        )

        #ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name= SPLIT_MODEL,
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)
        
        ic(len(source_text_chunks))

        translation_chunks = []
        for i in range(len(source_text_chunks)):
            # Will translate chunk i
            
            prompt = f"""You are a professional translation engine. Please translate the text delimited by triple backticks into Vietnamese without explanation.
            Original content:```你好```
            Translated content:Xin chào
            Original content:```{source_text_chunks[i]}```
            Translated content:"""    

            #ic(prompt)
            translation = get_completion_instruct_model(prompt)
            #ic(translation)
            translation_chunks.append(translation)
        
        return "".join(translation_chunks)
    
    
    
    
    prompt = f"""You are a professional translation engine. Please translate the text delimited by triple backticks into Vietnamese without explanation.
    Original content:```你好```
    Translated content:Xin chào
    Original content:```{source_text}```
    Translated content:"""
    
    ic("one_chunk_initial_translation_2")
    ic(source_text)    
    
    translation = get_completion_instruct_model(prompt)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
    )

    # 用户消息
    user_message = {"role": "user", "content": prompt}

    # 发送消息给 Claude，并获取响应
    try:
        response = claudclient.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[user_message],
            system=system_message  # 将系统信息作为顶级参数传递
        )
        
        reflection=response.content
        ic(reflection) # 打印 Claude 的响应
    except Exception as e:
        print(f"Error: {str(e)}")  # 错误处理
    
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    translation_2 = get_completion(prompt,system_message,model = SECOND_TRANSLATION_MODEL_2)

    return translation_2


def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    
    '''
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )
    '''
    
    #ic(source_text)
    #ic(len(source_text))
    
    ic("第一遍翻译")
    translation_1 = one_chunk_initial_translation_2(source_text)
    
    ic(len(translation_1))
    #ic(translation_1)
    
    # 将初始翻译结果写入文件
    with open('translation_1.txt', 'w', encoding='utf-8') as file:
        file.write(translation_1)
    
    ic("AI校对翻译")
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    # 将反思内容写入文件
    with open('reflection.txt', 'w', encoding='utf-8') as file:
        file.write(reflection)
    
    #ic(reflection)
    ic("根据校对第二遍翻译")
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )
    
    ic("########################")
    ic(len(translation_2))
    #ic(translation_2)
    
    # 将改进后的翻译结果写入文件
    with open('translation_2.txt', 'w', encoding='utf-8') as file:
        file.write(translation_2)

    
    return translation_2


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens

def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """
    
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """Translate the source_text from source_lang to target_lang."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name= SPLIT_MODEL,
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)
        
        ic(len(source_text_chunks))

        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        return "".join(translation_2_chunks)



if __name__ == "__main__":

    source_lang, target_lang, country = "Chinese", "Vietnamese", "Vietnam"
    
    #读取原文
    relative_path = "sample-texts/sourcetext.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    #读取翻译结果
    relative_path = "sample-texts/translationresult.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        translation_1 = file.read()
    
    reflection = one_chunk_reflect_on_translation(source_lang,target_lang,source_text,translation_1,country)
    ic(reflection)
