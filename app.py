import html
import os
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from transformers import pipeline

@st.cache_resource
def load_ner_model():
    """Загружает модель NER в оперативную память один раз при запуске."""
    hf_token = os.getenv("HF_TOKEN")
    return pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="simple",
        token=hf_token
    )

from app_back import (
    fetch_by_url,
    wiki_soup_2_content,
    wiki_content_2_text,
    wiki_content_2_links,
    file_2_ners,
    compare_2_files,
    ENCODING
)

st.set_page_config(page_title="Анализатор Текста", layout="wide")

st.title("📊 Сравнение статей Википедии и текстовых файлов")
st.write("Приложение скачивает статью из Википедии и сравнивает её с вашим файлом с помощью лемматизации и TF-IDF.")

col1, col2 = st.columns(2)

with col1:
    url = st.text_input(
        "Ссылка на Википедию:",
        value="https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5"
    )

with col2:
    uploaded_file = st.file_uploader("Загрузите файл для сравнения (любой текстовый формат)")


# Цветовая палитра для типов NE
NER_COLORS = {
    "PER": "#FFADAD",   # Красный / Розовый (Люди)
    "ORG": "#CAFFBF",   # Зеленый (Организации)
    "LOC": "#9BF6FF",   # Голубой (Локации)
    "MISC": "#FFD6A5"   # Оранжевый / Желтый (Разное)
}


def highlight_ners_in_text(text: str, entities: list) -> str:
    """
    Формирует HTML-строку с белым текстом на тёмном фоне и цветными плашками сущностей.
    """
    if not text:
        return '<div style="color: #888888;"><i>Текст отсутствует.</i></div>'
    if not entities:
        escaped_text = html.escape(text).replace("\n", "<br>")
        return f'<div style="font-size: 15px; line-height: 1.8; color: #FFFFFF; background-color: #1E1E1E; padding: 16px; border-radius: 8px;">{escaped_text}</div>'

    sorted_entities = sorted(entities, key=lambda x: x["start"])

    html_parts = []
    last_idx = 0

    for ent in sorted_entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity_group"]
        color = NER_COLORS.get(label, "#E0E0E0")

        # Экранируем и добавляем обычный текст до сущности
        html_parts.append(html.escape(text[last_idx:start]))

        # Плашка с текстом сущности (текст внутри плашки тёмный для контраста с пастельным фоном)
        ent_text = html.escape(text[start:end])
        badge = (
            f'<mark style="background-color: {color}; color: #111111; border-radius: 4px; padding: 2px 6px; '
            f'margin: 0 2px; display: inline-block; line-height: 1.4;">'
            f'<b>{ent_text}</b> '
            f'<span style="font-size: 0.7em; font-weight: bold; text-transform: uppercase; '
            f'opacity: 0.85; margin-left: 3px; color: #222222;">{label}</span>'
            f'</mark>'
        )
        html_parts.append(badge)
        last_idx = end

    # Добавляем оставшуюся часть текста
    html_parts.append(html.escape(text[last_idx:]))

    final_html = "".join(html_parts).replace("\n", "<br>")

    # Текст белого цвета на тёмно-сером фоне (#1E1E1E)
    return (
        f'<div style="font-size: 15px; line-height: 1.8; color: #FFFFFF; '
        f'background-color: #1E1E1E; padding: 18px; border-radius: 8px; '
        f'border: 1px solid #333333; max-height: 600px; overflow-y: auto;">'
        f'{final_html}'
        f'</div>'
    )

def get_gradient_color(val: float, metric_type: str = "cos"):
    if metric_type == "cos":
        norm_val = max(0.0, min(1.0, val / 0.65))
    else:
        norm_val = max(0.0, min(1.0, val / 0.25))

    if norm_val < 0.5:
        t = norm_val * 2
        r = int(220 + (255 - 220) * t)
        g = int(53 + (193 - 53) * t)
        b = int(69 + (7 - 69) * t)
    else:
        t = (norm_val - 0.5) * 2
        r = int(255 + (40 - 255) * t)
        g = int(193 + (167 - 193) * t)
        b = int(7 + (69 - 7) * t)

    return r, g, b


def render_metric_card(title: str, value: float, metric_type: str = "cos"):
    r, g, b = get_gradient_color(value, metric_type)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.22) 0%, rgba({r}, {g}, {b}, 0.05) 100%);
            border-left: 6px solid rgb({r}, {g}, {b});
            padding: 18px 22px;
            border-radius: 10px;
            margin-bottom: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);">
            <div style="font-size: 14px; font-weight: 600; color: #555; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 34px; font-weight: 800; color: rgb({r}, {g}, {b});">
                {value:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def convert_counter_to_csv(counter_dict):
    df = pd.DataFrame(counter_dict.items(), columns=["Лемма", "Количество"]).sort_values(by="Количество", ascending=False)
    return df, df.to_csv(index=False, encoding="utf-8-sig")


def create_wordcloud_fig(counter_dict, colormap="viridis"):
    if not counter_dict:
        return None

    height, width = 500, 1000
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    mask = ((y - center_y) ** 2 / (center_y * 0.85) ** 2) + ((x - center_x) ** 2 / (center_x * 0.85) ** 2) > 1
    mask = (mask * 255).astype(np.uint8)

    wc = WordCloud(
        width=width,
        height=height,
        background_color=None,
        mode="RGBA",
        colormap=colormap,
        max_words=65,
        prefer_horizontal=0.85,
        relative_scaling=0.4,
        collocations=False,
        mask=mask,
        margin=8
    ).generate_from_frequencies(counter_dict)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig




if st.button("🚀 Начать сравнение", type="primary"):
    if not url:
        st.error("Пожалуйста, укажите ссылку на Википедию.")
    elif not uploaded_file:
        st.error("Пожалуйста, загрузите второй файл для сравнения.")
    else:
        with st.spinner("Загрузка страницы Википедии, выделение сущностей и обработка текстов..."):
            ner_pipe = load_ner_model()
            soup = fetch_by_url(url)
            if not soup:
                st.error("Не удалось загрузить страницу по указанному URL.")
            else:
                cont = wiki_soup_2_content(soup)
                wiki_text = wiki_content_2_text(cont)

                wiki_links_counter = wiki_content_2_links(cont)

                raw_bytes = uploaded_file.getvalue()
                try:
                    user_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    user_text = raw_bytes.decode("cp1251", errors="ignore")

                f_wiki = tempfile.NamedTemporaryFile(mode="w+", encoding=ENCODING, delete=False)
                f_user = tempfile.NamedTemporaryFile(mode="w+", encoding=ENCODING, delete=False)

                try:
                    f_wiki.write(wiki_text)
                    f_user.write(user_text)
                    f_wiki.close()
                    f_user.close()

                    # Сравнение
                    results = compare_2_files(f_wiki.name, f_user.name)

                    # Извлечение текста и именованных сущностей
                    wiki_raw_text, wiki_ners = file_2_ners(f_wiki.name, ner_pipeline=ner_pipe)
                    user_raw_text, user_ners = file_2_ners(f_user.name, ner_pipeline=ner_pipe)

                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success("Сравнение успешно завершено!")

                        # Метрики
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            render_metric_card("Коэффициент Жаккара", results['jaccar'], metric_type="jaccard")
                        with m_col2:
                            render_metric_card("Косинусное сходство (TF-IDF)", results['cos_sim'], metric_type="cos")

                        st.divider()

                        # Вкладки
                        tab_overview, tab_wiki_lemmas, tab_user_lemmas, tab_wiki_links, tab_ners = st.tabs([
                            "🔍 Обзор сравнения",
                            "📚 Леммы из Википедии",
                            "📄 Леммы из вашего файла",
                            "🔗 Ссылки из Википедии",
                            "🏷️ Именованные сущности (NER)"
                        ])

                        # Вкладка 1: Пересечения
                        with tab_overview:
                            sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                                "Общие леммы (Пересечение)",
                                "Уникальные в Википедии",
                                "Уникальные в вашем файле"
                            ])
                            with sub_tab1:
                                st.write(f"Всего слов: **{len(results['intersection'])}**")
                                st.write(", ".join(sorted(results['intersection'])))
                            with sub_tab2:
                                st.write(f"Всего слов: **{len(results['l1_without_l2'])}**")
                                st.write(", ".join(sorted(results['l1_without_l2'])))
                            with sub_tab3:
                                st.write(f"Всего слов: **{len(results['l2_without_l1'])}**")
                                st.write(", ".join(sorted(results['l2_without_l1'])))

                        # Вкладка 2: Леммы Википедии
                        with tab_wiki_lemmas:
                            df_wiki, csv_wiki = convert_counter_to_csv(results["l1_counter"])

                            c1, c2 = st.columns([3, 1])
                            c1.markdown(f"Всего уникальных лемм: **{len(df_wiki)}** | Общее кол-во слов: **{sum(results['l1_counter'].values())}**")
                            c2.download_button(
                                label="📥 Скачать CSV (Википедия)",
                                data=csv_wiki,
                                file_name="wiki_lemmas.csv",
                                mime="text/csv"
                            )

                            st.subheader("☁️ Облако слов")
                            fig_wiki = create_wordcloud_fig(results["l1_counter"], colormap="Blues")
                            if fig_wiki:
                                st.pyplot(fig_wiki)

                            st.subheader("📊 Топ-15 самых частых лемм")
                            st.bar_chart(df_wiki.head(15), x="Лемма", y="Количество", color="#3182bd")

                            st.subheader("📋 Полный список лемм")
                            st.dataframe(df_wiki, use_container_width=True)

                        # Вкладка 3: Леммы пользователя
                        with tab_user_lemmas:
                            df_user, csv_user = convert_counter_to_csv(results["l2_counter"])

                            c1, c2 = st.columns([3, 1])
                            c1.markdown(f"Всего уникальных лемм: **{len(df_user)}** | Общее кол-во слов: **{sum(results['l2_counter'].values())}**")
                            c2.download_button(
                                label="📥 Скачать CSV (Ваш файл)",
                                data=csv_user,
                                file_name="user_lemmas.csv",
                                mime="text/csv"
                            )

                            st.subheader("☁️ Облако слов")
                            fig_user = create_wordcloud_fig(results["l2_counter"], colormap="Greens")
                            if fig_user:
                                st.pyplot(fig_user)

                            st.subheader("📊 Топ-15 самых частых лемм")
                            st.bar_chart(df_user.head(15), x="Лемма", y="Количество", color="#2ca02c")

                            st.subheader("📋 Полный список лемм")
                            st.dataframe(df_user, use_container_width=True)

                        # Вкладка 4: Ссылки из Википедии
                        with tab_wiki_links:
                            st.subheader("🔗 Внутренние ссылки из статьи Википедии")
                            if wiki_links_counter:
                                df_links = pd.DataFrame(
                                    wiki_links_counter.items(),
                                    columns=["Название статьи", "Количество упоминаний"]
                                ).sort_values(by="Количество упоминаний", ascending=False)

                                st.write(f"Всего уникальных ссылок: **{len(df_links)}**")
                                st.bar_chart(df_links.head(15), x="Название статьи", y="Количество упоминаний")
                                st.dataframe(df_links, use_container_width=True)
                            else:
                                st.info("В тексте статьи не найдено ссылок.")

                        # Вкладка 5: Именованные сущности с ЦВЕТНОЙ ПОДСВЕТКОЙ
                        with tab_ners:
                            st.subheader("🏷️ Выделение именованных сущностей в тексте (NER)")

                            # Легенда цветов
                            st.markdown(
                                """
                                <div style="display: flex; gap: 15px; background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
                                    <div><span style="background-color: #FFADAD; padding: 3px 8px; border-radius: 4px; font-weight: bold;">PER</span> Персоны</div>
                                    <div><span style="background-color: #CAFFBF; padding: 3px 8px; border-radius: 4px; font-weight: bold;">ORG</span> Организации</div>
                                    <div><span style="background-color: #9BF6FF; padding: 3px 8px; border-radius: 4px; font-weight: bold;">LOC</span> Локации</div>
                                    <div><span style="background-color: #FFD6A5; padding: 3px 8px; border-radius: 4px; font-weight: bold;">MISC</span> Прочее</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            ner_tab1, ner_tab2 = st.tabs(["📚 Текст Википедии", "📄 Ваш файл"])

                            with ner_tab1:
                                html_wiki_ner = highlight_ners_in_text(wiki_raw_text, wiki_ners)
                                st.markdown(html_wiki_ner, unsafe_allow_html=True)

                            with ner_tab2:
                                html_user_ner = highlight_ners_in_text(user_raw_text, user_ners)
                                st.markdown(html_user_ner, unsafe_allow_html=True)

                finally:
                    if os.path.exists(f_wiki.name):
                        os.remove(f_wiki.name)
                    if os.path.exists(f_user.name):
                        os.remove(f_user.name)