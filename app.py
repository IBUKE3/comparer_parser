# app.py
import os
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

from main import (
    fetch_by_url,
    wiki_soup_2_content,
    wiki_content_2_text,
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


def get_gradient_color(val: float, metric_type: str = "cos"):
    """
    Рассчитывает плавающий цвет RGB в градиенте Красный -> Желтый -> Зеленый.
    """
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
    """
    Отрисовывает чистую карточку с плавным градиентным фоном и значением.
    """
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
    df = pd.DataFrame(counter_dict.items(), columns=["Лемма", "Количество"]).sort_values(by="Количество",
                                                                                         ascending=False)
    return df, df.to_csv(index=False, encoding="utf-8-sig")


def create_wordcloud_fig(counter_dict, colormap="viridis"):
    """
    Генерирует стильное облако слов в форме овала с прозрачным фоном и высоким разрешением.
    """
    if not counter_dict:
        return None

    # Генерируем гладкую эллиптическую маску
    height, width = 500, 1000
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    # Формула эллипса для ограничения границ
    mask = ((y - center_y) ** 2 / (center_y * 0.85) ** 2) + ((x - center_x) ** 2 / (center_x * 0.85) ** 2) > 1
    mask = (mask * 255).astype(np.uint8)

    wc = WordCloud(
        width=width,
        height=height,
        background_color=None,  # Прозрачный фон
        mode="RGBA",  # Поддержка прозрачности
        colormap="viridis",  # Цветовая палитра
        max_words=65,  # Оптимальное количество слов без перегрузки
        prefer_horizontal=0.85,  # 85% слов по горизонтали (легко читать)
        relative_scaling=0.4,  # Сбалансированный масштаб шрифта
        collocations=False,
        mask=mask,
        margin=8
    ).generate_from_frequencies(counter_dict)

    # Создаем фигуру с высоким DPI для высокой четкости
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_alpha(0.0)  # Прозрачный холст Matplotlib
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
        with st.spinner("Загрузка страницы Википедии и обработка текстов..."):
            soup = fetch_by_url(url)
            if not soup:
                st.error("Не удалось загрузить страницу по указанному URL.")
            else:
                cont = wiki_soup_2_content(soup)
                wiki_text = wiki_content_2_text(cont)

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

                    results = compare_2_files(f_wiki.name, f_user.name)

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

                        tab_overview, tab_wiki_lemmas, tab_user_lemmas = st.tabs([
                            "🔍 Обзор сравнения",
                            "📚 Леммы из Википедии",
                            "📄 Леммы из вашего файла"
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
                            c1.markdown(
                                f"Всего уникальных лемм: **{len(df_wiki)}** | Общее кол-во слов: **{sum(results['l1_counter'].values())}**")
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

                        # Вкладка 3: Леммы из файла пользователя
                        with tab_user_lemmas:
                            df_user, csv_user = convert_counter_to_csv(results["l2_counter"])

                            c1, c2 = st.columns([3, 1])
                            c1.markdown(
                                f"Всего уникальных лемм: **{len(df_user)}** | Общее кол-во слов: **{sum(results['l2_counter'].values())}**")
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

                finally:
                    if os.path.exists(f_wiki.name):
                        os.remove(f_wiki.name)
                    if os.path.exists(f_user.name):
                        os.remove(f_user.name)