"""
Transliteration utilities for Russian <-> English text conversion.
Handles common transliteration patterns used in vehicle names.
"""

from typing import Optional
import re

# Russian to Latin transliteration mapping
RU_TO_LAT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
    'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
    'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
    'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
    'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
    'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch',
    'Ш': 'Sh', 'Щ': 'Shch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
    'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
}

# Latin to Russian transliteration mapping (common patterns)
LAT_TO_RU = {
    'shch': 'щ', 'sch': 'щ',
    'zh': 'ж', 'kh': 'х', 'ts': 'ц', 'ch': 'ч',
    'sh': 'ш', 'yu': 'ю', 'ya': 'я', 'yo': 'ё',
    'a': 'а', 'b': 'б', 'v': 'в', 'g': 'г', 'd': 'д',
    'e': 'е', 'z': 'з', 'i': 'и', 'y': 'й',
    'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
    'p': 'п', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'f': 'ф', 'h': 'х', 'c': 'ц', 'w': 'в', 'x': 'кс',
    'j': 'й', 'q': 'к'
}

# Common brand transliterations (Russian name -> English name)
BRAND_TRANSLATIONS = {
    # Russian -> English
    'мерседес': 'mercedes', 'мерс': 'mercedes',
    'бмв': 'bmw', 'бэха': 'bmw', 'бэмве': 'bmw',
    'ауди': 'audi',
    'фольксваген': 'volkswagen', 'фольц': 'volkswagen', 'вольксваген': 'volkswagen',
    'тойота': 'toyota', 'тоета': 'toyota',
    'хонда': 'honda', 'хёнда': 'honda',
    'ниссан': 'nissan', 'нисан': 'nissan',
    'мазда': 'mazda',
    'субару': 'subaru',
    'митсубиши': 'mitsubishi', 'митсубиси': 'mitsubishi', 'мицубиси': 'mitsubishi',
    'хёндай': 'hyundai', 'хёндэ': 'hyundai', 'хундай': 'hyundai', 'хендай': 'hyundai',
    'киа': 'kia',
    'форд': 'ford',
    'шевроле': 'chevrolet', 'шевролет': 'chevrolet',
    'опель': 'opel',
    'пежо': 'peugeot',
    'рено': 'renault', 'реналт': 'renault',
    'ситроен': 'citroen',
    'вольво': 'volvo',
    'шкода': 'skoda',
    'сеат': 'seat',
    'фиат': 'fiat',
    'альфа ромео': 'alfa romeo', 'альфа': 'alfa romeo',
    'порше': 'porsche', 'порш': 'porsche',
    'ягуар': 'jaguar',
    'лэнд ровер': 'land rover', 'ленд ровер': 'land rover',
    'рэйндж ровер': 'range rover', 'рендж ровер': 'range rover',
    'лексус': 'lexus',
    'инфинити': 'infiniti',
    'акура': 'acura',
    'кадиллак': 'cadillac',
    'линкольн': 'lincoln',
    'крайслер': 'chrysler',
    'додж': 'dodge',
    'джип': 'jeep',
    'лада': 'lada', 'ваз': 'lada',
    'газ': 'gaz',
    'уаз': 'uaz',
    'камаз': 'kamaz',
    'джили': 'geely',
    'чери': 'chery',
    'хавал': 'haval', 'хавейл': 'haval',
    'грейт волл': 'great wall',
}

# Reverse mapping
BRAND_TRANSLATIONS_REVERSE = {v: k for k, v in BRAND_TRANSLATIONS.items()}


def ru_to_latin(text: str) -> str:
    """Transliterate Russian text to Latin characters."""
    result = []
    for char in text:
        result.append(RU_TO_LAT.get(char, char))
    return ''.join(result)


def latin_to_ru(text: str) -> str:
    """Transliterate Latin text to Russian characters."""
    text = text.lower()
    result = []
    i = 0
    while i < len(text):
        # Try longer patterns first
        matched = False
        for length in [4, 3, 2, 1]:
            if i + length <= len(text):
                substring = text[i:i + length]
                if substring in LAT_TO_RU:
                    result.append(LAT_TO_RU[substring])
                    i += length
                    matched = True
                    break
        if not matched:
            result.append(text[i])
            i += 1
    return ''.join(result)


def normalize_brand(text: str) -> str:
    """
    Normalize brand name - convert common Russian brand names to English.
    Returns lowercase normalized version.
    """
    text_lower = text.lower().strip()
    
    # Check if it's a known Russian brand name
    if text_lower in BRAND_TRANSLATIONS:
        return BRAND_TRANSLATIONS[text_lower]
    
    return text_lower


def get_all_variants(text: str) -> list[str]:
    """
    Generate all possible variants of the text:
    - Original
    - Lowercase
    - Transliterated to Latin
    - Transliterated to Russian
    - Brand-normalized
    """
    variants = set()
    text_clean = text.strip()
    
    # Original and lowercase
    variants.add(text_clean)
    variants.add(text_clean.lower())
    
    # Transliterations
    variants.add(ru_to_latin(text_clean))
    variants.add(ru_to_latin(text_clean).lower())
    variants.add(latin_to_ru(text_clean))
    
    # Brand normalization
    normalized = normalize_brand(text_clean)
    variants.add(normalized)
    
    # Remove empty strings
    variants.discard('')
    
    return list(variants)


def detect_language(text: str) -> str:
    """Detect if text is primarily Russian or English."""
    russian_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    
    if russian_chars > latin_chars:
        return 'ru'
    elif latin_chars > russian_chars:
        return 'en'
    else:
        return 'unknown'


