"""
Vehicle Matching Pipeline
Fuzzy matching for vehicle marks and models with Russian/English support.
"""

from .pipeline import VehicleMatcher, find_similar_vehicles, MatchResult
from .transliteration import ru_to_latin, latin_to_ru, normalize_brand
from .fuzzy import FuzzyMatcher

__all__ = [
    'VehicleMatcher',
    'find_similar_vehicles', 
    'MatchResult',
    'FuzzyMatcher',
    'ru_to_latin', 
    'latin_to_ru', 
    'normalize_brand'
]
__version__ = '1.0.0'
