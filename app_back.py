import requests
from bs4 import BeautifulSoup
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from transformers import pipeline
from collections import Counter
from typing import Tuple
import math
import os
import tempfile


LANGUAGE = "russian"
ENCODING = "utf-8"



# Uncomment these two lines during the first run
# import nltk
# nltk.download('stopwords')

# could load all stopwords from a file. However, I prefer this way
STOP_WORDS = stopwords.words(LANGUAGE)
STOP_WORDS.extend(['в', 'на', 'под', 'над', 'за', 'у', 'о', 'об',
                   'обо', 'по', 'из', 'изо', 'ссо', 'от', 'до',
                   'для', 'к', 'ко', 'перед', 'при', 'через',
                   'без', 'около', 'про', 'после', 'во', 'и', 'а',
                   'но', 'да', 'или', 'либо', 'если', 'что', 'чтобы',
                   'когда', 'пока', 'хотя', 'будто', 'словно', 'ибо',
                   'потому что', 'так как', 'не', 'ни', 'ли', 'же',
                   'вот', 'вон', 'то', 'бы', 'б', 'ли', 'лишь',
                   'даже', 'ведь', 'дескать', 'мол', 'разве',
                   'неужели', 'исключительно', 'только', 'всё-таки',
                   'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
                   'мой', 'твой', 'свой', 'наш', 'ваш', 'кто', 'что',
                   'какой', 'чей', 'который', 'весь', 'вся', 'всё', 'все',
                   'этот', 'тот', 'такой', 'где', 'куда', 'откуда', 'когда',
                   'зачем', 'почему', 'как', 'сколько', 'также', 'тоже',
                   'здесь', 'там', 'тут', 'потом', 'затем', 'вообще', 'очень',
                   'более', 'менее', '!', '#', '$', '%', '&', "'", '*', '+',
                   '-', '/', '=', '?', '^', '_', '`', '{', '|', '}', '~', '.',
                   '(', ')', '"', '–', ',', ':', '«', '»', ';', '<', '>', '—', '...'])
SET_STOP = set(STOP_WORDS)
PM3 = MorphAnalyzer()


# Function returns BS object from given URL or False.
# Returns False, if an error occurred during fetching
def fetch_by_url(url: str) -> BeautifulSoup | bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 (WikiAnalyzerApp/1.0; mailto:your_email@example.com)"
    }
    try:
        r = requests.get(url, headers=headers) # processing with these headers. Else - 403 from wikipedia
        r.raise_for_status()

        # running this part, if there were no exceptions raised
        soup = BeautifulSoup(r.content, "html.parser")
        # print(soup.prettify())
        return soup

    except requests.exceptions.HTTPError as e:
        print("Возникла ошибка при попытке загрузить страницу. Ошибка:", e)
        return False
    except requests.exceptions.RequestException as e:
        print("Возникла ошибка при попытке загрузить страницу. Ошибка:", e)
        return False


# Returns the main part of the article without service parts - further as content
# Returns None if soup was incorrect or on an exception
def wiki_soup_2_content(soup: BeautifulSoup):
    if not soup:
        return None
    try:
        t = soup.find("div", id="mw-content-text")
        for s in ["Примечания", "Литература", "Комментарии", "Ссылки", "См._также"]:
            for elem in t.find_all("section"):
                if elem.attrs:
                    if "aria-labelledby" in elem.attrs:
                        if elem["aria-labelledby"] == s:
                            elem.decompose()
        return t
    except Exception as e:
        print("Произошла ошибка при парсинге на этапе удаления служебных частей")
        print(e)
        return None


# Turns content from wiki_soup_2_content(...) into usual text
# Takes all text from <p> and <li> tags and joins them into a single str
# Deletes links like [x] from text, where x is any
# Returns empty string if there was anything wrong with content
def wiki_content_2_text(t) -> str:
    if not t:
        return ""
    soup_tags = t.find_all(["p", "li"])
    res = ""
    for tag in soup_tags:
        s = re.sub(r"\[.*?]","",  tag.text)
        res += s + " "
    return res


# Returns Counter with all links to Wiki pages from content
# Returns empty Counter if there was anything wrong with content
# Links are searched in <p> and <li> tags
def wiki_content_2_links(t) -> Counter:
    c = Counter()
    for tag in t.find_all("p"):
        for a in tag.find_all("a"):
            if "title" in a.attrs:
                c[a["title"]]+=1
    for tag in t.find_all("li"):
        for a in tag.find_all("a"):
            if "title" in a.attrs:
                c[a["title"]]+=1
    return c


# Returns counter with number of tokens in the file
# Does not count tokens in SET_STOP - see global above
def file_2_tokens(file_name: str) -> Counter:
    c = Counter()
    try:
        with open(file_name, "r", encoding=ENCODING) as f:
            text = f.read().lower()
    except FileNotFoundError:
        print(f"Файл {file_name} не найден. Продолжаю работу, считая его пустым")
        return c
    tokens = word_tokenize(text)
    for i in tokens:
        if not (i in SET_STOP):
            c[i]+=1
    return c


# Gets file_name and prints named entities, recognised by wikineural
# app_back.py

# app_back.py

def chunk_text_with_offsets(text: str, max_chars: int = 1000) -> list[tuple[str, int]]:
    """
    Разбивает длинный текст на куски не более max_chars (не разрывая слова),
    возвращая список пар: (фрагмент_текста, начальное_смещение_в_символах).
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            # Ищем ближайший перенос строки или пробел, чтобы не разрезать слово
            space_pos = text.rfind('\n', start, end)
            if space_pos == -1 or space_pos <= start:
                space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                end = space_pos + 1

        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk, start))
        start = end

    return chunks

_NER_PIPELINE = None

def get_ner_pipeline():
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        _NER_PIPELINE = pipeline(
            "ner",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple"
        )
    return _NER_PIPELINE

# app_back.py

def file_2_ners(file_name: str, ner_pipeline=None) -> tuple[str, list]:
    try:
        with open(file_name, "r", encoding=ENCODING) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Файл {file_name} не найден.")
        return "", []

    if not text.strip():
        return text, []

    # Если pipeline не передан извне, загружаем локально
    if ner_pipeline is None:
        ner_pipeline = pipeline(
            "ner",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple"
        )

    all_entities = []
    chunks = chunk_text_with_offsets(text, max_chars=1000)

    for chunk_text, offset in chunks:
        chunk_entities = ner_pipeline(chunk_text)
        for ent in chunk_entities:
            ent["start"] += offset
            ent["end"] += offset
            all_entities.append(ent)

    return text, all_entities

# Gets tokens_counter from file_tokens
# Returns Counter of lemms
# Returns empty Counter if tokens_counter was empty
def tokens_counter_2_lemms(tokens_counter: Counter) -> Counter:
    r_c = Counter()
    for i in tokens_counter:
        r_c[PM3.parse(i)[0].normal_form] += tokens_counter[i]
    return r_c


# Applies tokenizer and lemmatizer to a file file_name
# Returns both counters as a tuple
def process_text_file(file_name: str) -> Tuple[Counter, Counter]:
    tokens_counter = file_2_tokens(file_name)
    lemms_counter = tokens_counter_2_lemms(tokens_counter)
    return tokens_counter, lemms_counter


# Gets 2 filenames and processes them
# Returns dictionary with following keys:
# "l1_counter": Counter of lemms in the first file
# "l2_counter": Counter of lemms in the second file
# "union": set of all lemms in both files (in any)
# "intersection": set of lemms that are in both files simultaneously
# "l1_without_l2": set of lemms that are in the first file, but not in the second
# "l2_without_l1": set of lemms that are in the second file, but not in the first
# "jaccar": jaccar coefficient
# "tf_idf1": tf_idf vector for the first file
# "tf_idf2": tf_idf2 vector for the second file
# "cos_sim": cos similarity of tf_idf1 and tf_idf2
def compare_2_files(file_1, file_2: str):
    t1, l1 = process_text_file(file_1)
    t2, l2 = process_text_file(file_2)
    set_l1 = set(l1)
    set_l2 = set(l2)
    intersection = set_l1.intersection(set_l2)
    union = set_l1.union(set_l2)
    l1_without_l2 = set_l1.difference(set_l2)
    l2_without_l1 = set_l2.difference(set_l1)
    difference = set_l1.symmetric_difference(set_l2)
    jaccar = len(intersection)/len(union)

    len1, len2 = 0, 0
    for word in l1:
        len1 += l1[word]
    for word in l2:
        len2 += l2[word]

    tf1, tf2, idf = {}, {}, {}
    tf_idf1, tf_idf2 = [], []
    if len1 and len2:
        for word in union:
            tf1[word] = l1[word]/len1
            tf2[word] = l2[word]/len2
            df = 0
            if word in l1:
                df += 1
            if word in l2:
                df += 1
            idf[word] = math.log((1+2)/(1+df)) + 1
            tf_idf1.append(tf1[word] * idf[word])
            tf_idf2.append(tf2[word] * idf[word])

    tf_idf_prod = sum(v1 * v2 for v1, v2 in zip(tf_idf1, tf_idf2))
    norm1 = math.sqrt(sum(v1 ** 2 for v1 in tf_idf1))
    norm2 = math.sqrt(sum(v2 ** 2 for v2 in tf_idf2))

    if norm1 == 0 or norm2 == 0:
        cos_sim = 0
    else:
        cos_sim = tf_idf_prod / (norm1 * norm2)

    return {
        "l1_counter": l1,
        "l2_counter": l2,
        "union": union,
        "intersection": intersection,
        "l1_without_l2": l1_without_l2,
        "l2_without_l1": l2_without_l1,
        "jaccar": jaccar,
        "tf_idf1": tf_idf1,
        "tf_idf2": tf_idf2,
        "cos_sim": cos_sim
    }
