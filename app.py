# app.py
import streamlit as st # Streamlit импорт должен быть одним из первых
import datetime
import sys
import os

# --- Убедитесь, что это ПЕРВАЯ команда Streamlit ---
st.set_page_config(page_title="Исторический ВестникЪ", layout="wide")

# --- Импорт основного модуля приложения ---
# Импортируем модули ПОСЛЕ set_page_config
try:
    # Импортируем функцию генерации и переменную для последней ошибки
    from modules.generator import generate_news, last_error
    from modules.models import NewsReport
    # Можно добавить необязательное сообщение об успехе, если нужно для отладки
    # st.sidebar.success("Модули 'generator' и 'models' импортированы.")
except ImportError as app_import_error:
    st.error(f"Критическая ошибка: Не удалось импортировать необходимые модули ('modules.generator' или 'modules.models').")
    st.error(f"Детали ошибки: {app_import_error}")
    st.error("Пожалуйста, проверьте структуру папок и наличие файлов `modules/generator.py` и `modules/models.py`.")
    # Выводим пути для помощи в отладке, если импорт не удался
    st.error(f"Текущая директория: {os.getcwd()}")
    st.error(f"Пути поиска Python: {sys.path}")
    st.stop() # Останавливаем приложение, так как без модулей оно неработоспособно
except Exception as app_other_error:
     st.error(f"Другая непредвиденная ошибка при импорте модулей: {app_other_error}")
     st.stop()


# --- Основной UI приложения ---
st.title("📜 Исторический ВестникЪ 📰")
st.caption("Генератор псевдо-исторических новостей на базе LLM")

# --- Ввод данных пользователем ---
col1, col2 = st.columns([1, 2])

with col1:
    default_date = datetime.date(1789, 7, 14)
    selected_date = st.date_input(
        "Выберите дату для выпуска газеты:",
        value=default_date,
        min_value=datetime.date(1000, 1, 1), # Ограничим разумно
        max_value=datetime.date.today()
    )
    # Формат для отображения и для передачи в функцию генерации
    selected_date_str = selected_date.strftime("%d %B %Y")

    # Опционально: выбор стиля эпохи
    era_options = ["XVIII", "XIX", "XVII", "XX"] # Добавьте нужные
    selected_era = st.selectbox("Стиль какого века предпочитаете?", era_options, index=0)

    num_articles = st.slider("Количество новостей в сводке:", min_value=1, max_value=5, value=3)

    generate_button = st.button("✨ Сгенерировать ВестникЪ!")


# --- Область вывода ---
with col2:
    st.subheader(f"Выпускъ отъ {selected_date_str} ({selected_era} вѣкъ)")

    if generate_button:
        # Убедимся еще раз, что функция доступна (хотя st.stop() выше должен был предотвратить это)
        if 'generate_news' in globals():
            with st.spinner(f"⏳ Редакція '{'Хронографъ'.upper()}' готовитъ свѣжій номеръ..."):
                # Вызов функции генерации
                news_report: NewsReport = generate_news(
                    target_date=selected_date_str,
                    era_style=selected_era,
                    num_articles=num_articles
                )

                # Отображение результата
                if news_report and news_report.articles:
                    st.success("📰 Свѣжій номеръ готовъ!")
                    # Отображение новостей
                    for i, article in enumerate(news_report.articles):
                        st.markdown(f"---")
                        st.markdown(f"### {i+1}. {article.headline}")
                        st.markdown(f"**Рубрика:** {article.rubric}")
                        st.markdown(f"*{article.date_location}*")
                        st.write(article.body)
                        st.caption(f"Репортажъ велъ: {article.reporter}")
                # Сообщение, если нет новостей, но и не было ошибки (проверяем last_error)
                # last_error импортируется из generator.py
                elif last_error is None:
                     st.info("Не удалось сгенерировать новости (возможно, нет данных или событий для этой даты).")
                # Если была ошибка, сообщения st.error/st.warning уже были выведены внутри generate_news

        else:
             # Это сообщение не должно появляться, если st.stop() сработал при ошибке импорта
             st.error("Критическая ошибка: функция генерации новостей недоступна.")

    else:
        st.info("Выберите дату и нажмите кнопку для генерации новостей.")


# --- Подвал ---
st.markdown("---")
st.caption("Создано с использованием LLM и Streamlit.")