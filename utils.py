import cv2
import numpy as np
import pytesseract
import re

def preprocess_image(image_pil):
    """Глубокая бинаризация для удаления фоновой сетки паспорта и бликов."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 1. Ограничение размера для стабильной работы памяти
    h, w = img.shape[:2]
    max_size = 1200
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 2. Перевод в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Удаление неравномерного освещения и вспышек (Top-Hat морфология)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gray = cv2.add(cv2.subtract(gray, tophat), blackhat)
    
    # 4. Адаптивное превращение в чистый ч/б (метод Оцу + Гаусс)
    # Это уберет водяные знаки и фоновый узор, оставив только черные буквы
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def postprocess_numeric_fields(text, mode="num"):
    """Исправление типичных ошибок OCR (путаница букв и цифр)."""
    # Замены для цифр (если OCR нашел букву там, где должна быть цифра)
    if mode == "num":
        replacements = {'О': '0', 'О': '0', 'З': '3', 'З': '3', 'Б': '6', 'Ч': '4', 'Т': '7', 'І': '1', 'I': '1', 'l': '1'}
        for char, digit in replacements.items():
            text = text.replace(char, digit)
        return re.sub(r'[^0-9\-\s\.]', '', text).strip()
    
    # Замены для текста (если OCR нашел цифру в ФИО)
    if mode == "text":
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'І'}
        for digit, char in replacements.items():
            text = text.replace(digit, char)
        return re.sub(r'[^а-яА-Я\s\-\.]', '', text).strip()
    return text

def extract_passport_data(image_pil):
    """Распознавание паспорта с помощью Tesseract OCR."""
    thresh_img = preprocess_image(image_pil)
    
    # Запуск Tesseract с поддержкой русского и английского (для машиночитаемой зоны)
    # --psm 11 ищет как можно больше текста на странице в любом порядке
    custom_config = r'--psm 11 -c preserve_interword_spaces=1'
    raw_text = pytesseract.image_to_string(thresh_img, lang='rus+eng', config=custom_config)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Шаблоны регулярных выражений
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}[\.\,\s]\d{2}[\.\,\s]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s]\d{3}')
    
    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
            
        # Ищем Код подразделения
        code_match = code_re.search(line)
        if code_match and passport_data["Код подразделения"] == "Не найдено":
            clean_code = postprocess_numeric_fields(code_match.group(), "num")
            if len(clean_code) >= 6:
                passport_data["Код подразделения"] = clean_code
                continue
                
        # Ищем Дату выдачи
        date_match = date_re.search(line)
        if date_match and passport_data["Дата выдачи"] == "Не найдено":
            clean_date = postprocess_numeric_fields(date_match.group(), "num").replace(' ', '.')
            passport_data["Дата выдачи"] = clean_date
            continue
            
        # Ищем Серию и номер
        num_match = passport_num_re.search(line)
        if num_match and passport_data["Серия и номер"] == "Не найдено":
            passport_data["Серия и номер"] = postprocess_numeric_fields(num_match.group(), "num")
            continue
            
        # Сборка ФИО (строки, состоящие преимущественно из русских букв верхнего регистра)
        clean_line = postprocess_numeric_fields(line, "text")
        if clean_line.isupper() and len(clean_line) > 4:
            # Исключаем служебные слова гознака
            if not any(word in clean_line.lower() for word in ["паспорт", "гражданина", "российской", "федерации"]):
                passport_data["ФИО / Текст"].append(clean_line)

    # Ограничиваем ФИО тремя строками (Фамилия, Имя, Отчество)
    passport_data["ФИО / Текст"] = passport_data["ФИО / Текст"][:3]
                
    return passport_data
