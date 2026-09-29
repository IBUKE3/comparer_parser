import requests
from bs4 import BeautifulSoup
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import os
import tempfile
from collections import Counter
from typing import Tuple
import math


LANGUAGE = "russian"
ENCODING = "utf-8"



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


# Returns the main part of the article without service tags
def wiki_soup_2_content(soup: BeautifulSoup):
    t = soup.find("div", id="mw-content-text")
    for s in ["Примечания", "Литература", "Комментарии", "Ссылки", "См._также"]:
        for elem in t.find_all("section"):
            if elem.attrs:
                if "aria-labelledby" in elem.attrs:
                    if elem["aria-labelledby"] == s:
                        elem.decompose()
    return t


def wiki_content_2_text(t) -> str:
    soup_tags = t.find_all(["p", "li"])
    res = ""
    for tag in soup_tags:
        s = re.sub(r"\[.*?]","",  tag.text)
        res += s + " "
    return res


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


def file_2_tokens(file_name: str) -> Counter:
    # Uncomment these two lines during the first run
    # import nltk
    # nltk.download('stopwords')

    stop_words = stopwords.words(LANGUAGE)
    stop_words.extend(['в', 'на', 'под', 'над', 'за', 'у', 'о', 'об',
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
                   '(', ')', '"', '–', ',', ':', '«', '»', ';', '<', '>', '—'])
    set_stop = set(stop_words)

    with open(file_name, "r", encoding=ENCODING) as f:
        text = f.read().lower()
    tokens = word_tokenize(text)
    c = Counter()
    for i in tokens:
        if not (i in set_stop):
            c[i]+=1
    return c


def tokens_counter_2_lemms(tokens_counter: Counter) -> Counter:
    r_c = Counter()
    pm3 = MorphAnalyzer()
    for i in tokens_counter:
        r_c[pm3.parse(i)[0].normal_form] += tokens_counter[i]
    return r_c


def process_text_file(file_name: str) -> Tuple[Counter, Counter]:
    tokens_counter = file_2_tokens(file_name)
    lemms_counter = tokens_counter_2_lemms(tokens_counter)
    return tokens_counter, lemms_counter


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
        "intersection": intersection,
        "l1_without_l2": l1_without_l2,
        "l2_without_l1": l2_without_l1,
        "jaccar": jaccar,
        "cos_sim": cos_sim
    }




'''
wiki_text_file_path = None

soup=fetch_by_url("https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5")
if soup:
    cont = wiki_soup_2_content(soup)
    with tempfile.NamedTemporaryFile(mode="w+", encoding=ENCODING, delete=False) as wiki_text_file:
        wiki_text_file.write(wiki_content_2_text(cont))
        wiki_text_file.flush()
        wiki_text_file_path = wiki_text_file.name
        wiki_text_file.seek(0)

        print(wiki_content_2_text(cont))

        # print("!!! TOKENS")
        # print(file_2_tokens(wiki_text_file.name))
        # print("!!! LEMMS")
        # print(tokens_counter_2_lemms(file_2_tokens(wiki_text_file.name)))


        res = compare_2_files(wiki_text_file.name, "some_text")

        print("______________________________________")
        print("Результаты сравнения двух файлов:")
        print("Пересечение множества лемм:", res["intersection"])
        print("Множество лемм первого файла без лемм из второго:", res["l1_without_l2"])
        print("Множество лемм второго файла без лемм из первого:", res["l2_without_l1"])
        print("Коэффициент Жаккара:", res["jaccar"])
        print("Косинусная близость TF-IDF векторов на основе лемм:", res["cos_sim"])
        print("Покрытие словарей друг другом:", res["intersection"])

        #compare_2_files(wiki_text_file.name, wiki_text_file.name)


    #print(wiki_content_2_links(cont))
else:
    print("OOps")

if wiki_text_file_path and os.path.exists(wiki_text_file_path):
        os.remove(wiki_text_file_path)


'''