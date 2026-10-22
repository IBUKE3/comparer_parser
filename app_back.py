import requests
from bs4 import BeautifulSoup
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from transformers import pipeline
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple
import math
import os


LANGUAGE = "russian"
ENCODING = "utf-8"



# Uncomment these two lines during the first run
# import nltk
# nltk.download('stopwords')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_words.txt"), "r", encoding=ENCODING) as f:
    SET_STOP = set(f.read().lower().strip().split(' '))
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


# Returns counter with number of tokens in the text
# Does not count tokens in SET_STOP - see global above
def str_2_tokens(text: str) -> Counter:
    c = Counter()
    tokens = word_tokenize(text.lower())
    for i in tokens:
        if not (i in SET_STOP):
            c[i] += 1
    return c


# Uses str_2_tokens to return tokens' counter from the file
def file_2_tokens(file_name: str) -> Counter:
    try:
        with open(file_name, "r", encoding=ENCODING) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Файл {file_name} не найден. Продолжаю работу, считая его пустым")
        return Counter()
    return str_2_tokens(text)


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
        )

    all_entities = []
    chunks = chunk_text_with_offsets(text, max_chars=1000)

    for chunk_text, offset in chunks:
        chunk_entities = ner_pipeline(chunk_text)
        for ent in chunk_entities:
            print(ent)
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


# Applies tokenizer and lemmatizer to a text as a string
# Returns both counters as a tuple
def process_str_text(text: str) -> Tuple[Counter, Counter]:
    tokens_counter = Counter()
    tokens = word_tokenize(text.lower())
    for i in tokens:
        if not (i in SET_STOP):
            tokens_counter[i]+=1
    lemms_counter = tokens_counter_2_lemms(tokens_counter)
    return tokens_counter, lemms_counter


# Applies tokenizer and lemmatizer to a file file_name
# Returns both counters as a tuple
def process_text_file(file_name: str) -> Tuple[Counter, Counter]:
    tokens_counter = file_2_tokens(file_name)
    lemms_counter = tokens_counter_2_lemms(tokens_counter)
    return tokens_counter, lemms_counter


# Gets two Counter of lemms
# Returns a tuple of their TF_IDF vectors
def calculate_tf_idf(l1, l2):
    len1, len2 = 0, 0
    for word in l1:
        len1 += l1[word]
    for word in l2:
        len2 += l2[word]
    print(len1)

    tf1, tf2, idf = {}, {}, {}
    tf_idf1, tf_idf2 = [], []
    if len1 and len2:
        for word in set(l1).union(set(l2)):
            tf1[word] = l1[word] / len1
            tf2[word] = l2[word] / len2
            df = 0
            if word in l1:
                df += 1
            if word in l2:
                df += 1
            idf[word] = math.log((1 + 2) / (1 + df)) + 1
            tf_idf1.append(tf1[word] * idf[word])
            tf_idf2.append(tf2[word] * idf[word])
    return (tf_idf1, tf_idf2)


# Returns cosine between v1 and v2
def calc_cos_sim(v1, v2):
    prod = sum(i1 * i2 for i1, i2 in zip(v1, v2))
    norm1 = math.sqrt(sum(i1 ** 2 for i1 in v1))
    norm2 = math.sqrt(sum(i2 ** 2 for i2 in v2))

    if norm1 == 0 or norm2 == 0:
        return 0
    else:
        return prod / (norm1 * norm2)


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
    tf_idf1, tf_idf2 = calculate_tf_idf(l1, l2)

    return {
        "l1_counter": l1,
        "l2_counter": l2,
        "union": set_l1.union(set_l2),
        "intersection": set_l1.intersection(set_l2),
        "l1_without_l2": set_l1.difference(set_l2),
        "l2_without_l1": set_l2.difference(set_l1),
        "jaccar": len(set_l1.intersection(set_l2))/len(set_l1.union(set_l2)),
        "tf_idf1": tf_idf1,
        "tf_idf2": tf_idf2,
        "cos_sim": calc_cos_sim(tf_idf1, tf_idf2)
    }


# Takes all lemms counters and returns one with all lems
# Value of c[word] is sum of all c_i[word] in list
def summarize_lemms_counters(list_lemmas_counters: list[Counter]):
    c = Counter()
    for l in list_lemmas_counters:
        c.update(l)
    return c


# DF as frequency of word as a key in lemms' counters
def calculate_df(list_lemmas_counters: list[Counter]):
    c = Counter()
    for l in list_lemmas_counters:
        c.update(l.keys())
    return c


# Using smoothed IDF here (better for 2 texts and generally)
def calculate_idf(general_lemmas_counter: Counter, df: Counter, n_texts: int):
    idf = {}
    for word in general_lemmas_counter.keys():
        idf[word]  = math.log((1 + n_texts) / (1 + df[word])) + 1
    return idf


# Gets list of lemmas' counters
# Returns lists of tf_idf vectors for every text
def calculate_tf_idf_multi(list_lemmas_counters: list[Counter]):
    n_texts = len(list_lemmas_counters)
    general_counter = summarize_lemms_counters(list_lemmas_counters)
    word_to_pos = {word: idx for idx, word in enumerate(general_counter)}
    n_lemms = len(general_counter)
    idf = calculate_idf(general_counter, calculate_df(list_lemmas_counters), n_texts)
    res = []
    for lc in list_lemmas_counters:
        subres = [0] * n_lemms
        words_in_text_n = sum(lc.values())
        if words_in_text_n > 0:
            for word in lc.keys():
                tf = lc[word]/words_in_text_n
                subres[word_to_pos[word]] = tf * idf[word]
        res.append(subres)
    return res


# Calculates cosine similarity between each 2 tf_idf vectors
# Builds symmetric matrix, where m[i,j]=m[j,i]=cos_sim(between tf_idf_i & tf_idf_j
def build_cosine_matrix(list_lemmas_counters: list[Counter]):
    n_texts = len(list_lemmas_counters)
    m = np.ones((n_texts, n_texts))
    tf_idfs = calculate_tf_idf_multi(list_lemmas_counters)
    for i in range(0, n_texts):
        for j in range(i+1, n_texts):
            r = calc_cos_sim(tf_idfs[i], tf_idfs[j])
            m[i, j] = r
            m[j, i] = r
    return m


# Builds symmetric matrix with jaccar' coef-s for each 2 texts
def build_jaccar_matrix(list_lemmas_counters: list[Counter]):
    n_texts = len(list_lemmas_counters)
    m = np.ones((n_texts, n_texts))
    sets_list = []
    for lc in list_lemmas_counters:
        sets_list.append(set(lc.keys()))
    for i in range(0, n_texts):
        for j in range(i+1, n_texts):
            united_set = sets_list[i].union(sets_list[j])
            u_len = len(united_set)
            if u_len == 0:
                r = 0
            else:
                r = len(sets_list[i].intersection(sets_list[j]))/u_len
            m[i, j] = r
            m[j, i] = r
    return m

# Builds matrices with cosine similarity and jaccar coefficient
# Returns them as pd.Dataframe
def calculate_corpus_matrices(list_lemmas_counters: list[Counter], list_names: list[str]):
    cos_matr = build_cosine_matrix(list_lemmas_counters)
    jac_matr = build_jaccar_matrix(list_lemmas_counters)
    df_cos = pd.DataFrame(cos_matr, index=list_names, columns=list_names)
    df_jac = pd.DataFrame(jac_matr, index=list_names, columns=list_names)
    return df_cos, df_jac