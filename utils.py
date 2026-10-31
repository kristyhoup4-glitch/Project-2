import cv2
import numpy as np
import easyocr
import re

# Инициализация OCR (при первом запуске веса скачиваются автоматически)
reader = easyocr.Reader(['ru', 'en'], gpu=False)

def preprocess_image(image_pil):
    """Качественная предобработка изображения для улучшения OCR."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Приведение к крупному стандартному размеру для стабильного чтения
    h, w = img.shape[:2]
    target_width = 1600
    if w != target_width:
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Повышение контраста через оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_text(text):
    """Очистка текста от мусорных символов OCR."""
    return text.replace('|', '').replace('I', '').replace('[', '').replace(']', '').strip()

def extract_passport_data(image_pil):
    """Распознавание данных с геометрической группировкой текста."""
    img = preprocess_image(image_pil)
    results = reader.readtext(img, detail=1)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Регулярные выражения
    passport_num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{4}\s?\d{6})')
    date_re = re.compile(r'\d{2}\s?[\.\,]\s?\d{2}\s?[\.\,]\s?\d{4}')
    code_re = re.compile(r'\d{3}\s?[\-\s]\s?\d{3}')

    other_lines = []
    
    for (bbox, text, prob) in results:
        if prob < 0.25:
            continue
            
        text = clean_text(text)
        if not text:
            continue
            
        # Поиск серии и номера
        num_match = passport_num_re.search(text)
        if num_match and passport_data["Серия и номер"] == "Не найдено":
            passport_data["Серия и номер"] = num_match.group()
            continue
            
        # Поиск кода подразделения
        code_match = code_re.search(text)
        if code_match and passport_data["Код подразделения"] == "Не найдено":
            passport_data["Код подразделения"] = code_match.group()
            continue

        # Поиск дат
        date_match = date_re.search(text)
        if date_match and passport_data["Дата выдачи"] == "Не найдено":
            passport_data["Дата выдачи"] = date_match.group().replace(',', '.')
            continue
            
        # Вычисление координат для сортировки
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_start = bbox[0][0]
        other_lines.append({"text": text, "y": y_center, "x": x_start})

    # Сортировка строк сверху вниз по экрану
    other_lines.sort(key=lambda item: item["y"])
    
    # Группировка слов в строках слева направо
    grouped_lines = []
    if other_lines:
        current_line = [other_lines[0]]
        for line in other_lines[1:]:
            if abs(line["y"] - current_line[-1]["y"]) < 20:
                current_line.append(line)
            else:
                current_line.sort(key=lambda item: item["x"])
                grouped_lines.append(" ".join([item["text"] for item in current_line]))
                current_line = [line]
        current_line.sort(key=lambda item: item["x"])
        grouped_lines.append(" ".join([item["text"] for item in current_line]))

    # Очистка от официальных штампов паспорта РФ
    for line in grouped_lines:
        lower_line = line.lower()
        if any(word in lower_line for word in ["паспорт", "гражданина", "российской", "федерации", "подпись"]):
            continue
        if len(line) > 3:
            passport_data["ФИО / Текст"].append(line)
            
    return passport_data
