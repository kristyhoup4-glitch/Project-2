import streamlit as st
from PIL import Image
from utils import extract_passport_data

st.set_page_config(page_title="Passport OCR AI", page_icon="🪪", layout="centered")

st.title("🪪 Распознавание паспортных данных")
st.write("Загрузите фотографию или скан главного разворота паспорта для автоматического извлечения текстовых данных нейросетью.")

# Компонент загрузки файла
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file)
    
    # Отображаем его на странице
    st.image(image, caption="Загруженный паспорт", use_container_width=True)
    
    st.write("---")
    
    with st.spinner("Нейросеть обрабатывает изображение..."):
        try:
            # Вызываем функцию распознавания
            parsed_data = extract_passport_data(image)
            
            # Выводим структурированные результаты
            st.success("Обработка успешно завершена!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Серия и номер", parsed_data["Серия и номер"])
                st.metric("Код подразделения", parsed_data["Код подразделения"])
            with col2:
                st.metric("Дата выдачи", parsed_data["Дата выдачи"])
            
            st.subheader("Найденные текстовые блоки (ФИО и организация):")
            for line in parsed_data["ФИО / Текст"]:
                st.write(f"- {line}")
                
        except Exception as e:
            st.error(f"Произошла ошибка при обработке: {e}")
