import cv2
import numpy as np
import easyocr
import re

# Инициализируем OCR в экономичном режиме памяти
reader = easyocr.Reader(['ru'], gpu=False, download_enabled=True)

def preprocess_image(image_pil):
    """Сжатие и базовая очистка изображения для экономии RAM сервера."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Жестко ограничиваем максимальный размер (спасает от ошибки Out of Memory)
    h, w = img.shape[:2]
    max_size = 1000
    if max(h, w) > max_size:
        if w > h:
            img = cv2.resize(img, (max_size, int(h * (max_size / w))), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (int(w * (max_size / h)), max_size), interpolation=cv2.INTER_AREA)
            
    # Переводим в градации серого и немного повышаем резкость
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def clean_text(text):
    """Очистка строк от мусора OCR."""
    text = re.sub(r'[^а-яА-Я0-9\s\.\,\-\<\>]', '', text) # Убираем английский мусор и спецсимволы
    return text.strip()

def extract_passport_data(image_pil):
    """Распознавание данных с защитой от утечки памяти."""
    img = preprocess_image(image_pil)
    results = reader.readtext(img, detail=1)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Сверхгибкие регулярные выражения (учитывают пробелы и опечатки OCR)
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}\s?[\.\,\s]\s?\d{2}\s?[\.\,\s]\s?\d{4}')
    code_re = re.compile(r'\d{3}\s?[\-\s]\s?\d{3}')

    raw_lines = []
    
    for (bbox, text, prob) in results:
        if prob < 0.20: # Игнорируем совсем блеклый шум
            continue
            
        text = clean_text(text)
        if len(text) < 2:
            continue
            
        # Проверяем спец-поля
        if passport_data["Серия и номер"] == "Не найдено" and passport_num_re.search(text):
            passport_data["Серия и номер"] = passport_num_re.search(text).group()
            continue
            
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(text):
            passport_data["Код подразделения"] = code_re.search(text).group()
            continue

        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(text):
            passport_data["Дата выдачи"] = date_re.search(text).group().replace(' ', '.')
            continue
            
        # Рассчитываем Y-координату для сортировки ФИО
        y_center = int((bbox[0][1] + bbox[2][1]) / 2)
        raw_lines.append((y_center, text))

    # Сортируем весь текстовый блок сверху вниз по документу
    raw_lines.sort(key=lambda x: x[0])
    
    # Фильтруем системные слова гознака
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "уфмс", "овд", "отделом", "отделение", "выдан"]
    
    for _, text in raw_lines:
        # Если строка состоит только из заглавных букв и длиннее 3 символов — это потенциально ФИО
        if text.isupper() and len(text) > 3:
            if not any(word in text.lower() for word in stop_words):
                passport_data["ФИО / Текст"].append(text)
        # Если ФИО не набралось, берем просто значимый текст
        elif len(passport_data["ФИО / Текст"]) < 3 and len(text) > 4:
             if not any(word in text.lower() for word in stop_words) and not text.isdigit():
                 passport_data["ФИО / Текст"].append(text)
                 
    return passport_data
