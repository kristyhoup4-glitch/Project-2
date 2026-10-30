import cv2
import numpy as np
import easyocr
import re

# Инициализируем OCR для русского и английского языков
# (При первом запуске модель скачается автоматически)
reader = easyocr.Reader(['ru', 'en'], gpu=False)

def preprocess_image(image_pil):
    """Предварительная обработка изображения для улучшения качества OCR."""
    image_np = np.array(image_pil)
    # Переводим в BGR для OpenCV
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Масштабируем, если картинка слишком маленькая
    h, w = img.shape[:2]
    if w < 1000:
        img = cv2.resize(img, (1000, int(h * (1000 / w))), interpolation=cv2.INTER_CUBIC)
    return img

def extract_passport_data(image_pil):
    """Основная функция распознавания данных."""
    img = preprocess_image(image_pil)
    
    # Запускаем распознавание текста нейросетью
    results = reader.readtext(img, detail=1)
    
    # Переменные для хранения найденных полей
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Регулярные выражения для фильтрации данных
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}\.\d{2}\.\d{4}')
    code_re = re.compile(r'\d{3}-\d{3}')

    for (bbox, text, prob) in results:
        text = text.strip()
        if prob < 0.35: # Игнорируем неуверенно распознанный текст
            continue
            
        # Ищем серию и номер
        if passport_num_re.search(text) and passport_data["Серия и номер"] == "Не найдено":
            passport_data["Серия и номер"] = passport_num_re.search(text).group()
            continue
            
        # Ищем код подразделения
        if code_re.search(text) and passport_data["Код подразделения"] == "Не найдено":
            passport_data["Код подразделения"] = code_re.search(text).group()
            continue

        # Ищем даты (выдачи или рождения)
        if date_re.search(text):
            if passport_data["Дата выдачи"] == "Не найдено":
                passport_data["Дата выдачи"] = date_re.search(text).group()
            continue
            
        # Всё остальное (ФИО, кем выдан) собираем в общий массив для вывода
        if len(text) > 2 and not text.isdigit():
            passport_data["ФИО / Текст"].append(text)
            
    return passport_data
