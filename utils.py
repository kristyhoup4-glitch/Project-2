import cv2
import numpy as np
import easyocr
import re

# Инициализируем нейросеть EasyOCR с включенными продвинутыми алгоритмами детекции блоков (CRAFT)
reader = easyocr.Reader(['ru'], gpu=False)

def preprocess_maximum_precision(image_pil):
    """Супер-контрастная обработка для проявления скрытых цифр и красного шрифта."""
    image_np = np.array(image_pil)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 1. НЕ сжимаем, а нормализуем картинку до ОЧЕНЬ высокого разрешения,
    # чтобы мелкие цифры стали крупными для нейросети
    h, w = img.shape[:2]
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, (2000, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
    # 2. Разделяем каналы, чтобы усилить красный цвет серии и номера
    b, g, r = cv2.split(img)
    # Красный шрифт на светлом фоне лучше всего контрастирует в зеленом (g) и синем (b) каналах
    enhanced_gray = cv2.addWeighted(b, 0.5, g, 0.5, 0)
    
    # 3. Локальное выравнивание контрастности (CLAHE) с увеличенным порогом
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    final_gray = clahe.apply(enhanced_gray)
    
    return cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)

def fix_ocr_errors(text, is_numeric=False):
    """Исправление путаницы символов OCR на уровне частых опечаток документов РФ."""
    text = text.strip().upper()
    # Убираем явный мусор, оставляя буквы, цифры и разделители
    text = re.sub(r'[^А-Я0-9\.\,\-\s\/\\]', '', text)
    
    if is_numeric:
        replacements = {
            'О': '0', 'о': '0', 'З': '3', 'з': '3', 'Б': '6', 'б': '6', 
            'Ч': '4', 'ч': '4', 'Т': '7', 'т': '7', 'І': '1', 'I': '1', 
            'L': '1', 'S': '5', 'А': '4', 'А': '4'
        }
        for char, digit in replacements.items():
            text = text.replace(char, digit)
    else:
        replacements = {'0': 'О', '3': 'З', '4': 'Ч', '1': 'И'}
        for digit, char in replacements.items():
            text = text.replace(digit, char)
            
    return text.strip()

def extract_passport_data(image_pil):
    """Глобальное сканирование документа на максимальных настройках нейросети."""
    # Получаем супер-контрастное изображение высокого разрешения
    processed_img = preprocess_maximum_precision(image_pil)
    
    # Запускаем EasyOCR с параметрами максимального поиска:
    # paragraph=False (ищет отдельные слова, не сливая строки)
    # adjust_contrast=0.7 (внутреннее усиление контраста нейросети)
    results = reader.readtext(processed_img, detail=1, paragraph=False, adjust_contrast=0.7)
    
    passport_data = {
        "Серия и номер": "Не найдено",
        "Дата выдачи": "Не найдено",
        "Код подразделения": "Не найдено",
        "ФИО / Текст": []
    }
    
    # Сверхгибкие регулярные выражения, учитывающие любые разделители (пробелы, точки, слеши, тире)
    # Для серии и номера: ищет 10 цифр подряд или группы 2+2+6 цифр
    num_re = re.compile(r'(\d{2}\s?\d{2}\s?\d{6})|(\d{10})|(\d{4}\s?\d{6})')
    # Для даты: формат ДД.ММ.ГГГГ с любыми разделителями
    date_re = re.compile(r'\d{2}[\.\,\-\s\/\\_]\d{2}[\.\,\-\s\/\\_]\d{4}')
    # Для кода подразделения: 3 цифры, разделитель, 3 цифры
    code_re = re.compile(r'\d{3}[\-\s\/\\_]\d{3}')
    
    raw_lines = []
    
    # Пробегаемся по всем найденным нейросетью блокам
    for (bbox, text, prob) in results:
        if prob < 0.10 or len(text.strip()) < 2:  # Минимальный порог, чтобы не пропустить блеклые цифры
            continue
            
        # Очищаем края строки
        clean_line = text.strip("[]|()_~`—- ")
        
        # 1. Пробуем найти Дату выдачи
        num_cleaned = fix_ocr_errors(clean_line, is_numeric=True)
        if passport_data["Дата выдачи"] == "Не найдено" and date_re.search(num_cleaned):
            raw_date = date_re.search(num_cleaned).group()
            # Стандартизируем разделители в точки
            passport_data["Дата выдачи"] = re.sub(r'[\,\-\s\/\\_]', '.', raw_date)
            continue
            
        # 2. Пробуем найти Код подразделения
        if passport_data["Код подразделения"] == "Не найдено" and code_re.search(num_cleaned):
            raw_code = code_re.search(num_cleaned).group()
            # Стандартизируем разделитель в дефис
            passport_data["Код подразделения"] = re.sub(r'[\s\/\\_]', '-', raw_code)
            continue

        # 3. Пробуем найти Серию и Номер
        if passport_data["Серия и номер"] == "Не найдено" and num_re.search(num_cleaned):
            passport_data["Серия и номер"] = num_re.search(num_cleaned).group()
            continue
            
        # Сохраняем координату Y для сортировки ФИО
        y_coord = int(bbox[0][1])
        raw_lines.append((y_coord, clean_line))
        
    # Сортируем оставшийся текст сверху вниз
    raw_lines.sort(key=lambda x: x[0])
    
    stop_words = ["паспорт", "гражданина", "российской", "федерации", "россии", "подпись", "личная", "код", "выдан", "подразделения"]
    
    # Собираем ФИО (как в прошлый раз — этот блок работает отлично)
    for _, text in raw_lines:
        text_fixed = fix_ocr_errors(text, is_numeric=False)
        text_only_letters = re.sub(r'[^А-Яа-я\s\-]', '', text_fixed).strip()
        
        if len(text_only_letters) > 3:
            if not any(word in text_only_letters.lower() for word in stop_words):
                if text_only_letters.isupper() or len(passport_data["ФИО / Текст"]) < 3:
                    passport_data["ФИО / Текст"].append(text_only_letters)
                    
    passport_data["ФИО / Текст"] = list(dict.fromkeys(passport_data["ФИО / Текст"]))[:3]
    
    # Красивое форматирование серии/номера пробелами (45 12 123456)
    if passport_data["Серия и номер"] != "Не найдено":
        digits_only = re.sub(r'[^0-9]', '', passport_data["Серия и номер"])
        if len(digits_only) == 10:
            passport_data["Серия и номер"] = f"{digits_only[:2]} {digits_only[2:4]} {digits_only[4:]}"
            
    return passport_data
