"""
Fuzzy matching strategies for vehicle name matching.
Combines multiple algorithms for robust matching with typos.
"""

from typing import Callable
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


class FuzzyMatcher:
    """
    Multi-strategy fuzzy matcher that combines different algorithms
    for robust matching against typos and variations.
    """
    
    def __init__(self, weights: dict[str, float] | None = None):
        """
        Initialize fuzzy matcher with optional custom weights.
        
        Args:
            weights: Dictionary mapping strategy names to their weights.
                     Default weights are optimized for vehicle name matching.
        """
        self.weights = weights or {
            'ratio': 0.25,           # Basic Levenshtein ratio
            'partial': 0.20,         # Partial string matching (for substrings)
            'token_sort': 0.20,      # Token-sorted comparison
            'token_set': 0.15,       # Token set comparison (handles word order)
            'jaro_winkler': 0.20,    # Good for typos at the end of words
        }
    
    def ratio(self, s1: str, s2: str) -> float:
        """Basic Levenshtein ratio (0-100)."""
        return fuzz.ratio(s1, s2)
    
    def partial_ratio(self, s1: str, s2: str) -> float:
        """Partial string matching - good when one string contains another."""
        return fuzz.partial_ratio(s1, s2)
    
    def token_sort_ratio(self, s1: str, s2: str) -> float:
        """Token-sorted ratio - handles word order differences."""
        return fuzz.token_sort_ratio(s1, s2)
    
    def token_set_ratio(self, s1: str, s2: str) -> float:
        """Token set ratio - handles extra/missing words."""
        return fuzz.token_set_ratio(s1, s2)
    
    def jaro_winkler(self, s1: str, s2: str) -> float:
        """Jaro-Winkler similarity - gives more weight to prefix matches."""
        return JaroWinkler.similarity(s1, s2) * 100
    
    def combined_score(self, s1: str, s2: str) -> float:
        """
        Calculate combined weighted score from all strategies.
        
        Returns:
            Score from 0 to 100, where 100 is perfect match.
        """
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()
        
        # Quick exact match check
        if s1_lower == s2_lower:
            return 100.0
        
        scores = {
            'ratio': self.ratio(s1_lower, s2_lower),
            'partial': self.partial_ratio(s1_lower, s2_lower),
            'token_sort': self.token_sort_ratio(s1_lower, s2_lower),
            'token_set': self.token_set_ratio(s1_lower, s2_lower),
            'jaro_winkler': self.jaro_winkler(s1_lower, s2_lower),
        }
        
        weighted_sum = sum(
            scores[key] * self.weights.get(key, 0) 
            for key in scores
        )
        
        return weighted_sum
    
    def get_detailed_scores(self, s1: str, s2: str) -> dict[str, float]:
        """
        Get detailed breakdown of all matching scores.
        Useful for debugging and tuning.
        """
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()
        
        return {
            'ratio': self.ratio(s1_lower, s2_lower),
            'partial_ratio': self.partial_ratio(s1_lower, s2_lower),
            'token_sort_ratio': self.token_sort_ratio(s1_lower, s2_lower),
            'token_set_ratio': self.token_set_ratio(s1_lower, s2_lower),
            'jaro_winkler': self.jaro_winkler(s1_lower, s2_lower),
            'combined': self.combined_score(s1, s2),
        }


def quick_ratio(s1: str, s2: str) -> float:
    """Quick ratio calculation for filtering candidates."""
    return fuzz.ratio(s1.lower(), s2.lower())


def prefix_match_score(s1: str, s2: str, prefix_len: int = 3) -> float:
    """
    Score based on prefix matching.
    Good for catching typos where the start is correct.
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()
    
    if not s1_lower or not s2_lower:
        return 0.0
    
    min_len = min(len(s1_lower), len(s2_lower), prefix_len)
    
    if s1_lower[:min_len] == s2_lower[:min_len]:
        # Bonus for matching prefix
        return 30.0 + fuzz.ratio(s1_lower, s2_lower) * 0.7
    
    return fuzz.ratio(s1_lower, s2_lower)

