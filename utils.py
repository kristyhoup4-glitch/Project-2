import cv2
import numpy as np
import easyocr
import re
import streamlit as st

@st.cache_resource
def load_ocr_model():
    """Ленивая загрузка модели ИИ в кэш Streamlit для защиты от падения сервера по памяти."""
    return easyocr.Reader(['ru'], gpu=False)

def preprocess_maximum_precision(image_pil):
    """Оптимальная контрастная обработка для проявления цифр и красного шрифта."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Ограничиваем размер до разумного максимума, чтобы не взорвать ОЗУ при обработке
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_AREA)
        
    # Разделяем каналы, чтобы усилить красный цвет серии и номера
    b, g, r = cv2.split(img)
    enhanced_gray = cv2.addWeighted(b, 0.5, g, 0.5, 0)
    
    # Локальное выравнивание контрастности (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    final_gray = clahe.apply(enhanced_gray)
    
    return cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)

def fix_ocr_errors(text, is_numeric=False):
    """Исправление путаницы символов OCR на уровне частых опечаток документов РФ."""
    text = text.strip().upper()
    text = re.sub(r'[^А-Я0-9\.\,\-\s\/\\]', '', text)
    
    if is_numeric:
        replacements = {
            'О': '0', 'о': '0', 'З': '3', 'з': '3', 'Б': '6', 'б': '6', 
            'Ч': '4', 'ч': '4', 'Т': '7', 'т': '7', 'І': '1', 'I': '1', 
            'L': '1', 'S': '5', 'А': '4'
        }
        for char, digit in replacements.items():
            text = text.replace(char, digit)
    else:
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for digit, char in replacements.items():
            text = text.replace(digit, char)
            
    return text.strip()

def extract_passport_data(image_pil):
    """Глобальное сканирование документа с вызовом модели из кэша."""
    # Получаем модель ИИ из безопасного кэша
    reader = load_ocr_model()
    
    processed_img = preprocess_maximum_precision(image_pil)
    results = reader.readtext(processed_img, detail=1, paragraph=False, adjust_contrast=0.5)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Регулярные выражения
    num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{10})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}[\.\,\-\s\/\\_]\d{2}[\.\,\-\s\/\\_]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s\/\\_]\d{3}')
    
    raw_lines = []
    
    for (bbox, text, prob) in results:
        if prob < 0.12 or len(text.strip()) < 2:
            continue
            
        clean_line = text.strip("[]|()_~`—- ")
        num_cleaned = fix_ocr_errors(clean_line, is_numeric=True)
        
        # 1. Поиск Даты выдачи
        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(num_cleaned):
            raw_date = date_re.search(num_cleaned).group()
            passport_data["Дата выдачи"] = re.sub(r'[\,\-\s\/\\_]', '.', raw_date)
            continue
            
        # 2. Поиск Кода подразделения
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(num_cleaned):
            raw_code = code_re.search(num_cleaned).group()
            passport_data["Код подразделения"] = re.sub(r'[\s\/\\_]', '-', raw_code)
            continue

        # 3. Поиск Серии и Номера
        if passport_data["Серия и номер"] == "Не найдено" and num_re.search(num_cleaned):
            passport_data["Серия и номер"] = num_re.search(num_cleaned).group()
            continue
            
        # Исправленное безопасное извлечение координаты Y
        try:
            y_coord = int(bbox[0][1])
        except (IndexError, TypeError, ValueError):
            y_coord = 0
            
        raw_lines.append((y_coord, clean_line))
        
    # Сортируем текст сверху вниз
    raw_lines.sort(key=lambda x: x[0])
    
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "россии", "подпись", "личная", "код", "выдан", "подразделения"]
    
    for _, text in raw_lines:
        text_fixed = fix_ocr_errors(text, is_numeric=False)
        text_only_letters = re.sub(r'[^А-Яа-я\s\-]', '', text_fixed).strip()
        
        if len(text_only_letters) > 3:
            if not any(word in text_only_letters.lower() for word in stop_words):
                if text_only_letters.isupper() or len(passport_data["ФИО / Текст"]) < 3:
                    passport_data["ФИО / Текст"].append(text_only_letters)
                    
    passport_data["ФИО / Текст"] = list(dict.fromkeys(passport_data["ФИО / Текст"]))[:3]
    
    # Красивое форматирование серии/номера пробелами
    if passport_data["Серия и номер"] != "Не найдено":
        digits_only = re.sub(r'[^0-9]', '', passport_data["Серия и номер"])
        if len(digits_only) == 10:
            passport_data["Серия и номер"] = f"{digits_only[:2]} {digits_only[2:4]} {digits_only[4:]}"
            
    return passport_data
