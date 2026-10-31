import cv2
import numpy as np
import easyocr
import re

# Инициализируем нейросеть EasyOCR (только русский язык для экономии памяти)
reader = easyocr.Reader(['ru'], gpu=False)

def preprocess_image(image_pil):
    """Качественное улучшение картинки: убираем шумы и фоновую сетку гознака."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 1. Приводим к фиксированной ширине для стабильности нейросети
    h, w = img.shape[:2]
    target_width = 1000
    scale = target_width / w
    img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 2. Повышаем локальный контраст текста через адаптивное выравнивание (CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. Мягкая фильтрация размытием для склейки разрывов в буквах
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

def fix_ocr_errors(text, is_numeric=False):
    """Принудительное исправление путаницы букв и цифр (О/0, З/3, И/1)."""
    text = text.strip()
    if is_numeric:
        # Если это номер, код или дата — превращаем ошибочные буквы в цифры
        replacements = {'О': '0', 'о': '0', 'З': '3', 'з': '3', 'Б': '6', 'б': '6', 'Ч': '4', 'ч': '4', 'Т': '7', 'т': '7', 'І': '1', 'I': '1', 'l': '1'}
        for char, digit in replacements.items():
            text = text.replace(char, digit)
        # Убираем все случайные не-цифровые символы (кроме точек и дефисов)
        return re.sub(r'[^0-9\.\,\-\s]', '', text).strip()
    else:
        # Если это ФИО — превращаем случайные цифры обратно в буквы
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for digit, char in replacements.items():
            text = text.replace(digit, char)
        return text

def extract_passport_data(image_pil):
    """Главная функция извлечения текстовых блоков паспорта."""
    img = preprocess_image(image_pil)
    
    # Запускаем распознавание текста нейросетью
    results = reader.readtext(img, detail=1)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Строгие и точные регулярные выражения для документов РФ
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}[\.\,\-\s]\d{2}[\.\,\-\s]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s]\d{3}')
    
    raw_lines = []
    
    # Шаг 1: Первичный разбор и фильтрация по регулярным выражениям
    for (bbox, text, prob) in results:
        if prob < 0.15 or len(text.strip()) < 2:
            continue
            
        clean_line = text.strip("[]|()_~`—")
        
        # Проверяем на серию и номер паспорта
        num_candidate = fix_ocr_errors(clean_line, is_numeric=True)
        if passport_data["Серия и номер"] == "Не найдено" and passport_num_re.search(num_candidate):
            passport_data["Серия и номер"] = passport_num_re.search(num_candidate).group()
            continue
            
        # Проверяем на код подразделения (например, 770-001)
        code_candidate = fix_ocr_errors(clean_line, is_numeric=True)
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(code_candidate):
            passport_data["Код подразделения"] = code_re.search(code_candidate).group()
            continue

        # Проверяем на дату выдачи (например, 12.12.2012)
        date_candidate = fix_ocr_errors(clean_line, is_numeric=True)
        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(date_candidate):
            # Приводим дату к красивому стандартному виду с точками
            passport_data["Дата выдачи"] = re.sub(r'[\,\-\s]', '.', date_re.search(date_candidate).group())
            continue
            
        # Если это не спец-поле, сохраняем для ФИО (берем безопасную координату Y из первой точки bbox)
        try:
            y_coord = int(bbox[0][1])
        except Exception:
            y_coord = 0
        raw_lines.append((y_coord, clean_line))
        
    # Шаг 2: Сортируем все текстовые строки сверху вниз (по координате Y)
    raw_lines.sort(key=lambda x: x[0])
    
    # Мусорные слова-исключения, которые часто встречаются в паспортах
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "уфмс", "овд", "отделом", "отделение", "выдан", "код", "подразделения"]
    
    # Шаг 3: Сборка и фильтрация ФИО
    for _, text in raw_lines:
        text_fixed = fix_ocr_errors(text, is_numeric=False)
        # Оставляем только русские буквы, дефисы и пробелы
        text_only_letters = re.sub(r'[^а-яА-Я\s\-]', '', text_fixed).strip()
        
        if len(text_only_letters) > 3:
            # Отсекаем служебные надписи Гознака
            if not any(word in text_only_letters.lower() for word in stop_words):
                # Если строка в верхнем регистре (КАПС) — это 100% ФИО
                if text_only_letters.isupper() or len(passport_data["ФИО / Текст"]) < 3:
                    passport_data["ФИО / Текст"].append(text_only_letters)
                    
    # Убираем дубликаты строк и берем первые три (Фамилия, Име, Отчество)
    passport_data["ФИО / Текст"] = list(dict.fromkeys(passport_data["ФИО / Текст"]))[:3]
    return passport_data
