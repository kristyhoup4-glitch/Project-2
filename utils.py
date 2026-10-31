import cv2
import numpy as np
import easyocr
import re

# Инициализация нейросети в режиме распознавания отдельных слов (параграфный режим отключен для точности)
reader = easyocr.Reader(['ru'], gpu=False)

def align_and_crop_passport(img):
    """Сверхточный поиск текстовых контуров для выравнивания документа."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Размытие и выделение границ Кэнни
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    # Ищем контуры самого крупного объекта (паспорта)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        large_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(large_cnt) > 50000:
            rect = cv2.minAreaRect(large_cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Вырезаем и выравниваем по углу наклона
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))
            
            # Разворачиваем в альбомную ориентацию, если вырезалось вертикально
            if height > width:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return warped
    return img

def preprocess_for_ocr(roi):
    """Финальная подготовка вырезанной зоны (бинаризация) перед OCR."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Морфологическое укрупнение шрифта для блеклых принтеров
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Адаптивный порог контраста
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def clean_and_format(text, mode="text"):
    """Жесткая фильтрация символов в зависимости от типа поля."""
    text = text.strip().upper()
    if mode == "num":
        # Исправляем частые путаницы OCR букв и цифр
        replacements = {'О': '0', 'З': '3', 'Б': '6', 'Ч': '4', 'Т': '7', 'І': '1', 'I': '1', 'L': '1', 'S': '5'}
        for c, d in replacements.items():
            text = text.replace(c, d)
        return re.sub(r'[^0-9\-\.\s]', '', text).strip()
    else:
        # Для ФИО убираем цифры и мусор
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for d, c in replacements.items():
            text = text.replace(d, c)
        return re.sub(r'[^А-Я\s\-]', '', text).strip()

def extract_passport_data(image_pil):
    """Разделение паспорта на логические зоны (ROI) и их поочередное чтение."""
    # Конвертация PIL в OpenCV
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 1. Нормализация размера кадра
    h, w = img.shape[:2]
    target_w = 1200
    img = cv2.resize(img, (target_w, int(h * (target_w / w))), interpolation=cv2.INTER_CUBIC)
    
    # 2. Умное выравнивание геометрии документа
    img = align_and_crop_passport(img)
    h, w = img.shape[:2]
    
    # 3. Деление паспорта на жесткие процентные зоны (ROI)
    # Координаты выверены по ГОСТу бланка паспорта РФ
    roi_top_info = img[0:int(h*0.45), 0:w]       # Кем выдан, дата, код
    roi_fio = img[int(h*0.45):int(h*0.80), 0:w]   # Фамилия, Имя, Отчество
    roi_num = img[0:h, int(w*0.82):w]             # Серия и номер (вертикальная строка справа)

    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Шаблоны регулярных выражений
    date_re = re.compile(r'\d{2}[\.\,\s]\d{2}[\.\,\s]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s]\d{3}')
    num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{10})|(\d{4}\s?\d{6})')

    # --- ЧТЕНИЕ ЗОНЫ 1: Верхний блок (Даты и коды) ---
    res_top = reader.readtext(preprocess_for_ocr(roi_top_info), detail=0)
    for text in res_top:
        cleaned = clean_and_format(text, "num")
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(cleaned):
            passport_data["Код подразделения"] = code_re.search(cleaned).group()
        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(cleaned):
            passport_data["Дата выдачи"] = date_re.search(cleaned).group().replace(' ', '.').replace(',', '.')

    # --- ЧТЕНИЕ ЗОНЫ 2: Вертикальный блок (Серия и Номер) ---
    # Поворачиваем боковую зону на 90 градусов, чтобы OCR прочитал её горизонтально
    roi_num_rotated = cv2.rotate(roi_num, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res_num = reader.readtext(preprocess_for_ocr(roi_num_rotated), detail=0)
    for text in res_num:
        cleaned = clean_and_format(text, "num")
        if num_re.search(cleaned):
            passport_data["Серия и номер"] = num_re.search(cleaned).group()
            break

    # --- ЧТЕНИЕ ЗОНЫ 3: Центральный блок (ФИО) ---
    res_fio = reader.readtext(roi_fio, detail=0) # ФИО лучше читать без агрессивного thresh
    fio_lines = []
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "россии", "подпись", "личная"]
    
    for text in res_fio:
        cleaned = clean_and_format(text, "text")
        if len(cleaned) > 2 and cleaned.isupper():
            if not any(word in cleaned.lower() for word in stop_words):
                fio_lines.append(cleaned)
                
    # Оставляем первые 3 уникальные строки (Фамилия, Имя, Отчество)
    passport_data["ФИО / Текст"] = list(dict.fromkeys(fio_lines))[:3]
    
    # Если боковой номер не считался, ищем его по всему документу в качестве подстраховки
    if passport_data["Серия и номер"] == "Не найдено":
        for text in res_top:
            cleaned = clean_and_format(text, "num")
            if num_re.search(cleaned):
                passport_data["Серия и номер"] = num_re.search(cleaned).group()

    return passport_data
