import cv2
import numpy as np
import easyocr
import re

# Инициализируем нейросеть EasyOCR (только русский язык для экономии RAM)
reader = easyocr.Reader(['ru'], gpu=False)

def preprocess_image(image_pil):
    """Базовое выравнивание размера изображения."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    target_w = 1200
    img = cv2.resize(img, (target_w, int(h * (target_w / w))), interpolation=cv2.INTER_CUBIC)
    return img

def enhance_for_digits(roi):
    """Мягкий контраст специально для мелких цифр, дат и красного шрифта серии."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_and_format(text, mode="text"):
    """Коррекция опечаток OCR на стыке букв и цифр."""
    text = text.strip().upper()
    if mode == "num":
        replacements = {'О': '0', 'о': '0', 'З': '3', 'з': '3', 'Б': '6', 'б': '6', 'Ч': '4', 'ч': '4', 'Т': '7', 'т': '7', 'І': '1', 'I': '1', 'L': '1', 'S': '5'}
        for c, d in replacements.items():
            text = text.replace(c, d)
        return re.sub(r'[^0-9\-\.\,\s]', '', text).strip()
    else:
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for d, c in replacements.items():
            text = text.replace(d, c)
        return re.sub(r'[^А-Я\s\-]', '', text).strip()

def extract_passport_data(image_pil):
    """Двухуровневое извлечение данных: по зонам + глобальный поиск для подстраховки."""
    img = preprocess_image(image_pil)
    h, w = img.shape[:2]
    
    # 1. Выделяем логические зоны (процентное соотношение по ГОСТу паспорта РФ)
    roi_top_info = img[0:int(h*0.50), 0:w]       # Кем выдан, дата, код подразделения
    roi_fio = img[int(h*0.42):int(h*0.82), 0:w]   # Зона ФИО
    roi_num = img[0:h, int(w*0.80):w]             # Вертикальная полоса с серией/номером справа

    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Регулярные выражения
    date_re = re.compile(r'\d{2}[\.\,\-\s]\d{2}[\.\,\-\s]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s]\d{3}')
    num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})|(\d{10})')

    # --- ЭТАП 1: РАСПОЗНАВАНИЕ ФИО (Работает идеально) ---
    res_fio = reader.readtext(roi_fio, detail=0)
    fio_lines = []
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "россии", "подпись", "личная", "код", "выдан"]
    
    for text in res_fio:
        cleaned = clean_and_format(text, "text")
        if len(cleaned) > 2 and cleaned.isupper():
            if not any(word in cleaned.lower() for word in stop_words):
                fio_lines.append(cleaned)
    passport_data["ФИО / Текст"] = list(dict.fromkeys(fio_lines))[:3]

    # --- ЭТАП 2: РАСПОЗНАВАНИЕ КОДА И ДАТЫ (С мягким улучшением картинок) ---
    res_top = reader.readtext(enhance_for_digits(roi_top_info), detail=0)
    for text in res_top:
        cleaned = clean_and_format(text, "num")
        
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(cleaned):
            passport_data["Код подразделения"] = code_re.search(cleaned).group()
            
        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(cleaned):
            raw_date = date_re.search(cleaned).group()
            passport_data["Дата выдачи"] = re.sub(r'[\,\-\s]', '.', raw_date)

    # --- ЭТАП 3: РАСПОЗНАВАНИЕ СЕРИИ И НОМЕРА (Поворот + Глобальный поиск) ---
    # Пытаемся прочитать из боковой вертикальной зоны, повернув её
    roi_num_rotated = cv2.rotate(roi_num, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res_num = reader.readtext(enhance_for_digits(roi_num_rotated), detail=0)
    for text in res_num:
        cleaned = clean_and_format(text, "num")
        if num_re.search(cleaned):
            passport_data["Серия и номер"] = num_re.search(cleaned).group()
            break

    # БРОНЕБОЙНАЯ ПОДСТРАХОВКА: Если серия/номер или код все еще не найдены, 
    # сканируем ВЕСЬ документ целиком без деления на зоны
    if passport_data["Серия и номер"] == "Не найдено" or passport_data["Код подразделения"] == "Не найдено" or passport_data["Дата выдачи"] == "Не найдено":
        res_global = reader.readtext(enhance_for_digits(img), detail=0)
        for text in res_global:
            cleaned = clean_and_format(text, "num")
            
            if passport_data["Серия и номер"] == "Не найдено" and num_re.search(cleaned):
                passport_data["Серия и номер"] = num_re.search(cleaned).group()
                
            if passport_data["Код подразделения"] == "Не найдено" and code_re.search(cleaned):
                passport_data["Код подразделения"] = code_re.search(cleaned).group()
                
            if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(cleaned):
                raw_date = date_re.search(cleaned).group()
                passport_data["Дата выдачи"] = re.sub(r'[\,\-\s]', '.', raw_date)

    # Финальное форматирование серии и номера для красоты (добавляем пробелы: ХХ ХХ ХХХХХХ)
    if passport_data["Серия и номер"] != "Не найдено":
        digits_only = re.sub(r'\s', '', passport_data["Серия и номер"])
        if len(digits_only) == 10:
            passport_data["Серия и номер"] = f"{digits_only[:2]} {digits_only[2:4]} {digits_only[4:]}"

    return passport_data
