import cv2
import numpy as np
import easyocr
import re

# Инициализация OCR (скачивание моделей произойдет автоматически при первом вызове)
reader = easyocr.Reader(['ru', 'en'], gpu=False)

def preprocess_image(image_pil):
    """Продвинутая предобработка для улучшения качества OCR."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 1. Приведение к крупному стандартному размеру
    h, w = img.shape[:2]
    target_width = 1600
    if w != target_width:
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # 2. Перевод в оттенки серого и адаптивное повышение контраста (CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Возвращаем цветное изображение, но OCR лучше кормить качественным серым/контрастным
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_text(text):
    """Очистка текста от типичных артефактов OCR."""
    text = text.replace('|', '').replace('I', '').replace('[', '').replace(']', '')
    return text.strip()

def extract_passport_data(image_pil):
    """Функция распознавания с группировкой строк по геометрии."""
    img = preprocess_image(image_pil)
    results = reader.readtext(img, detail=1)
    
    # Структура для итоговых данных
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Регулярные выражения с поддержкой возможных опечаток OCR
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}\s?[\.\,]\s?\d{2}\s?[\.\,]\s?\d{4}')
    code_re = re.compile(r'\d{3}\s?[\-\s]\s?\d{3}')

    # Сначала вытаскиваем строго форматированные данные (номера, коды, даты)
    other_lines = []
    
    for (bbox, text, prob) in results:
        if prob < 0.25:  # Отсекаем совсем нечитаемый шум
            continue
            
        text = clean_text(text)
        if not text:
            continue
            
        # 1. Проверка на серию и номер
        num_match = passport_num_re.search(text)
        if num_match and passport_data["Серия и number"] == "Не найдено":
            passport_data["Серия и номер"] = num_match.group()
            continue
            
        # 2. Проверка на код подразделения
        code_match = code_re.search(text)
        if code_match and passport_data["Код подразделения"] == "Не найдено":
            passport_data["Код подразделения"] = code_match.group()
            continue

        # 3. Проверка на даты
        date_match = date_re.search(text)
        if date_match and passport_data["Дата выдачи"] == "Не найдено":
            passport_data["Дата выдачи"] = date_match.group().replace(',', '.')
            continue
            
        # Если это обычный текст (ФИО или Кем выдан), сохраняем вместе с координатами для сортировки
        # bbox выглядит как: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        y_center = (bbox[0][1] + bbox[2][1]) / 2  # Находим центр блока по вертикали
        x_start = bbox[0][0]
        other_lines.append({"text": text, "y": y_center, "x": x_start})

    # Сортируем оставшийся текст сверху вниз (по координате Y)
    # Это склеит разорванные строки ФИО и "Кем выдан" в правильном порядке чтения
    other_lines.sort(key=lambda item: item["y"])
    
    # Группируем слова, которые находятся на одной горизонтальной линии (плюс-минус 15 пикселей)
    grouped_lines = []
    if other_lines:
        current_line = [other_lines[0]]
        for line in other_lines[1:]:
            if abs(line["y"] - current_line[-1]["y"]) < 20: # Порог объединения в одну строку
                current_line.append(line)
            else:
                current_line.sort(key=lambda item: item["x"]) # Сортируем слова слева направо
                grouped_lines.append(" ".join([item["text"] for item in current_line]))
                current_line = [line]
        current_line.sort(key=lambda item: item["x"])
        grouped_lines.append(" ".join([item["text"] for item in current_line]))

    # Фильтруем мусорные короткие строки
    for line in grouped_lines:
        # Убираем строки, состоящие из служебных слов паспорта, оставляя только персональные данные
        lower_line = line.lower()
        if any(word in lower_line for word in ["паспорт", "гражданина", "российской", "федерации", "подпись"]):
            continue
        if len(line) > 3:
            passport_data["ФИО / Текст"].append(line)
            
    return passport_data

