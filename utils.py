import cv2
import numpy as np
import easyocr
import re

# Инициализируем нейросеть EasyOCR (только русский язык для экономии RAM)
reader = easyocr.Reader(['ru'], gpu=False)

def preprocess_image(image_pil):
    """Мягкое выравнивание размера и контраста без разрушения букв."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Задаем оптимальную ширину кадра для EasyOCR
    h, w = img.shape[:2]
    target_width = 1200
    scale = target_width / w
    img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Легкое размытие для удаления цифрового шума камеры
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return blurred

def fix_ocr_errors(text, is_numeric=False):
    """Коррекция частых опечаток нейросети при чтении букв и цифр."""
    text = text.strip()
    if is_numeric:
        # Если поле должно состоять из цифр, меняем похожие буквы
        replacements = {'О': '0', 'О': '0', 'З': '3', 'З': '3', 'Б': '6', 'Ч': '4', 'Т': '7', 'І': '1', 'I': '1', 'l': '1'}
        for char, digit in replacements.items():
            text = text.replace(char, digit)
        return text
    else:
        # Если поле текстовое (ФИО), убираем цифры-опечатки
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for digit, char in replacements.items():
            text = text.replace(digit, char)
        return text

def extract_passport_data(image_pil):
    """Извлечение данных с мягкой фильтрацией."""
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
    date_re = re.compile(r'\d{2}[\.\,\-\s]\d{2}[\.\,\-\s]\d{4}')
    code_re = re.compile(r'\d{3}[\-\s]\d{3}')
    
    raw_lines = []
    
    for (bbox, text, prob) in results:
        if prob < 0.20 or len(text.strip()) < 2:
            continue
            
        # Очищаем текст от мусорных знаков препинания по краям
        clean_line = text.strip("[]|()_")
        
        # 1. Поиск серии и номера
        # Проверяем строку как с буквами (для исправления опечаток), так и чистую цифровую
        fixed_num_text = fix_ocr_errors(clean_line, is_numeric=True)
        num_match = passport_num_re.search(fixed_num_text)
        if num_match and passport_data["Серия и номер"] == "Не найдено":
            passport_data["Серия и номер"] = num_match.group()
            continue
            
        # 2. Поиск кода подразделения
        fixed_code_text = fix_ocr_errors(clean_line, is_numeric=True)
        code_match = code_re.search(fixed_code_text)
        if code_match and passport_data["Код подразделения"] == "Не найдено":
            passport_data["Код подразделения"] = code_match.group()
            continue

        # 3. Поиск дат
        fixed_date_text = fix_ocr_errors(clean_line, is_numeric=True)
        date_match = date_re.search(fixed_date_text)
        if date_match and passport_data["Дата выдачи"] == "Не найдено":
            passport_data["Дата выдачи"] = date_match.group().replace(' ', '.').replace(',', '.')
            continue
            
        # Сохраняем координату Y для ФИО
        y_center = int((bbox + bbox) / 2)
        raw_lines.append((y_center, clean_line))
        
    # Сортируем текст сверху вниз
    raw_lines.sort(key=lambda x: x)
    
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "уфмс", "овд", "отделом", "отделение", "выдан", "код"]
    
    for _, text in raw_lines:
        text_fixed = fix_ocr_errors(text, is_numeric=False)
        # Убираем цифры и спецсимволы из ФИО
        text_only_letters = re.sub(r'[^а-яА-Я\s\-]', '', text_fixed).strip()
        
        if len(text_only_letters) > 3:
            if not any(word in text_only_letters.lower() for word in stop_words):
                # Если строка написана капсом — с высокой вероятностью это ФИО в паспорте
                if text_only_letters.isupper() or len(passport_data["ФИО / Текст"]) < 3:
                    passport_data["ФИО / Текст"].append(text_only_letters)
                    
    passport_data["ФИО / Текст"] = list(dict.fromkeys(passport_data["ФИО / Текст"]))[:3]
    return passport_data
