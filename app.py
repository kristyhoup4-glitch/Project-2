import streamlit as st
from PIL import Image
from utils import extract_passport_data

# Базовая конфигурация страницы
st.set_page_config(page_title="Passport OCR AI", page_icon="🪪", layout="centered")
st.html("<html lang='en' class='notranslate' translate='no'>") 

st.title("🪪 Распознавание паспортных данных")
st.write("Загрузите фотографию или скан главного разворота паспорта.")

# Компонент загрузки файла работает независимо от ИИ
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженный паспорт", use_column_width=True)
    st.write("---")
    
    # Модель ИИ активируется строго внутри индикатора spinner по запросу
    with st.spinner("Нейросеть обрабатывает изображение... Пожалуйста, подождите."):
        try:
            parsed_data = extract_passport_data(image)
            st.success("Обработка успешно завершена!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Серия и номер", parsed_data["Серия и номер"])
                st.metric("Код подразделения", parsed_data["Код подразделения"])
            with col2:
                st.metric("Дата выдачи", parsed_data["Дата выдачи"])
            
            st.subheader("📋 Персональные данные:")
            fio = parsed_data["ФИО / Текст"]
            
            st.text_input("Фамилия", fio[0] if len(fio) > 0 else "Не найдено")
            st.text_input("Имя", fio[1] if len(fio) > 1 else "Не найдено")
            st.text_input("Отчество", fio[2] if len(fio) > 2 else "Не найдено")
                
        except Exception as e:
            st.error(f"Произошла ошибка при обработке: {e}")
