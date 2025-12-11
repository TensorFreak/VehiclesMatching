"""
Main vehicle matching pipeline.
Loads vehicle data and provides fuzzy matching for user queries.
"""

import pandas as pd
from dataclasses import dataclass
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

from .fuzzy import FuzzyMatcher
from .transliteration import (
    ru_to_latin, 
    normalize_brand, 
    get_all_variants,
    detect_language,
    BRAND_TRANSLATIONS
)


@dataclass
class MatchResult:
    """Result of a vehicle match."""
    mark: str
    model: str
    mark_score: float
    model_score: float
    combined_score: float
    original_index: int
    
    def __repr__(self):
        return f"MatchResult(mark='{self.mark}', model='{self.model}', score={self.combined_score:.1f})"


@dataclass
class BrandMatch:
    """Result of a brand match with its models."""
    brand: str
    brand_score: float
    models: list[tuple[str, float]]  # List of (model_name, model_score)
    
    def __repr__(self):
        return f"BrandMatch(brand='{self.brand}', score={self.brand_score:.1f}, models={len(self.models)})"


@dataclass 
class HierarchicalResult:
    """Result from hierarchical matching."""
    brand: str
    model: str
    brand_score: float
    model_score: float
    combined_score: float
    rank_brand: int      # Rank of brand in brand search (1-based)
    rank_model: int      # Rank of model within brand (1-based)
    original_index: int
    
    def __repr__(self):
        return (f"HierarchicalResult('{self.brand}' #{self.rank_brand} -> "
                f"'{self.model}' #{self.rank_model}, score={self.combined_score:.1f})")


class VehicleMatcher:
    """
    Fuzzy matcher for vehicle marks and models.
    
    Loads a parquet file with 'mark' and 'model' columns and provides
    fast fuzzy matching against user queries with typos.
    
    Supports Russian and English input with automatic transliteration.
    """
    
    def __init__(
        self, 
        parquet_path: str | None = None, 
        df: pd.DataFrame | None = None,
        mark_column: str = 'mark',
        model_column: str = 'model'
    ):
        """
        Initialize the matcher.
        
        Args:
            parquet_path: Path to parquet file with mark and model columns.
            df: Alternatively, pass a DataFrame directly.
            mark_column: Name of the mark/brand column (default: 'mark', also accepts 'brand').
            model_column: Name of the model column (default: 'model').
        """
        self.fuzzy = FuzzyMatcher()
        
        if parquet_path:
            self.df = pd.read_parquet(parquet_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either parquet_path or df must be provided")
        
        # Auto-detect column names
        if mark_column not in self.df.columns:
            if 'brand' in self.df.columns:
                mark_column = 'brand'
            elif 'mark' in self.df.columns:
                mark_column = 'mark'
            else:
                raise ValueError("DataFrame must have 'mark' or 'brand' column")
        
        if model_column not in self.df.columns:
            raise ValueError(f"DataFrame must have '{model_column}' column")
        
        self.mark_column = mark_column
        self.model_column = model_column
        
        # Preprocess data
        self._build_index()
    
    def _build_index(self):
        """Build lookup indexes for fast matching."""
        # Clean and normalize data
        self.df['mark_clean'] = self.df[self.mark_column].fillna('').astype(str).str.lower().str.strip()
        self.df['model_clean'] = self.df[self.model_column].fillna('').astype(str).str.lower().str.strip()
        
        # Create combined field for full matching
        self.df['full_name'] = self.df['mark_clean'] + ' ' + self.df['model_clean']
        
        # Build unique marks index with transliterations
        self.unique_marks = self.df['mark_clean'].unique().tolist()
        self.mark_to_indices = {}
        for idx, row in self.df.iterrows():
            mark = row['mark_clean']
            if mark not in self.mark_to_indices:
                self.mark_to_indices[mark] = []
            self.mark_to_indices[mark].append(idx)
        
        # Build mark variants for better matching
        self.mark_variants = {}  # variant -> original mark
        for mark in self.unique_marks:
            variants = get_all_variants(mark)
            for v in variants:
                self.mark_variants[v] = mark
    
    def _normalize_query(self, text: str) -> list[str]:
        """
        Normalize query text and generate variants for matching.
        Returns list of variants to try.
        """
        text = text.strip()
        if not text:
            return []
        
        variants = [text.lower()]
        
        # Add transliterated variant
        variants.append(ru_to_latin(text).lower())
        
        # Check if it's a known brand name
        normalized = normalize_brand(text)
        if normalized != text.lower():
            variants.append(normalized)
        
        # Check brand translations
        text_lower = text.lower()
        if text_lower in BRAND_TRANSLATIONS:
            variants.append(BRAND_TRANSLATIONS[text_lower])
        
        return list(set(variants))
    
    def find_similar_marks(
        self, 
        query_mark: str, 
        top_k: int = 5,
        min_score: float = 50.0
    ) -> list[tuple[str, float]]:
        """
        Find similar marks to the query.
        
        Args:
            query_mark: User's mark query (can have typos).
            top_k: Number of top results to return.
            min_score: Minimum score threshold (0-100).
            
        Returns:
            List of (mark, score) tuples sorted by score descending.
        """
        query_variants = self._normalize_query(query_mark)
        
        if not query_variants:
            return []
        
        # Check for exact matches first (including brand translations)
        for variant in query_variants:
            if variant in self.mark_variants:
                return [(self.mark_variants[variant], 100.0)]
        
        # Fuzzy match against all unique marks
        all_scores = {}
        
        for variant in query_variants:
            # Use rapidfuzz's process.extract for fast matching
            results = process.extract(
                variant,
                self.unique_marks,
                scorer=fuzz.WRatio,  # Weighted ratio - good for typos
                limit=top_k * 2
            )
            
            for mark, score, _ in results:
                if mark not in all_scores or score > all_scores[mark]:
                    all_scores[mark] = score
        
        # Sort by score and filter
        sorted_results = sorted(all_scores.items(), key=lambda x: -x[1])
        filtered = [(m, s) for m, s in sorted_results if s >= min_score]
        
        return filtered[:top_k]
    
    def find_similar_models(
        self,
        query_model: str,
        mark: str | None = None,
        top_k: int = 5,
        min_score: float = 50.0
    ) -> list[tuple[str, float]]:
        """
        Find similar models to the query.
        
        Args:
            query_model: User's model query (can have typos).
            mark: If provided, only search models for this mark.
            top_k: Number of top results to return.
            min_score: Minimum score threshold (0-100).
            
        Returns:
            List of (model, score) tuples sorted by score descending.
        """
        query_variants = self._normalize_query(query_model)
        
        if not query_variants:
            return []
        
        # Get models to search
        if mark:
            mark_lower = mark.lower().strip()
            if mark_lower in self.mark_to_indices:
                indices = self.mark_to_indices[mark_lower]
                models = self.df.loc[indices, 'model_clean'].unique().tolist()
            else:
                # Try fuzzy match on mark first
                similar_marks = self.find_similar_marks(mark, top_k=1)
                if similar_marks:
                    best_mark = similar_marks[0][0]
                    indices = self.mark_to_indices.get(best_mark, [])
                    models = self.df.loc[indices, 'model_clean'].unique().tolist()
                else:
                    models = self.df['model_clean'].unique().tolist()
        else:
            models = self.df['model_clean'].unique().tolist()
        
        # Fuzzy match
        all_scores = {}
        
        for variant in query_variants:
            results = process.extract(
                variant,
                models,
                scorer=fuzz.WRatio,
                limit=top_k * 2
            )
            
            for model, score, _ in results:
                if model not in all_scores or score > all_scores[model]:
                    all_scores[model] = score
        
        sorted_results = sorted(all_scores.items(), key=lambda x: -x[1])
        filtered = [(m, s) for m, s in sorted_results if s >= min_score]
        
        return filtered[:top_k]
    
    def match(
        self,
        query_mark: str,
        query_model: str,
        top_k: int = 5,
        min_score: float = 50.0,
        mark_weight: float = 0.4,
        model_weight: float = 0.6
    ) -> list[MatchResult]:
        """
        Find the most similar mark+model combinations.
        
        This is the main matching function. It:
        1. Finds candidate marks similar to query_mark
        2. For each candidate mark, finds similar models
        3. Combines scores and returns top results
        
        Args:
            query_mark: User's mark query (e.g., "мерседес", "mersedes", "Mercedes").
            query_model: User's model query (e.g., "е класс", "e-class").
            top_k: Number of top results to return.
            min_score: Minimum combined score threshold (0-100).
            mark_weight: Weight for mark score in combined score.
            model_weight: Weight for model score in combined score.
            
        Returns:
            List of MatchResult objects sorted by combined score.
        """
        results = []
        
        # Find candidate marks
        similar_marks = self.find_similar_marks(query_mark, top_k=10, min_score=40.0)
        
        if not similar_marks:
            # If no marks found, search all
            similar_marks = [(m, 50.0) for m in self.unique_marks[:100]]
        
        # For each candidate mark, find similar models
        for mark, mark_score in similar_marks:
            similar_models = self.find_similar_models(
                query_model, 
                mark=mark, 
                top_k=5,
                min_score=40.0
            )
            
            for model, model_score in similar_models:
                combined = mark_weight * mark_score + model_weight * model_score
                
                if combined >= min_score:
                    # Get original row index
                    mask = (self.df['mark_clean'] == mark) & (self.df['model_clean'] == model)
                    indices = self.df[mask].index.tolist()
                    original_idx = indices[0] if indices else -1
                    
                    # Get original (non-lowercased) values
                    if original_idx >= 0:
                        original_mark = self.df.loc[original_idx, self.mark_column]
                        original_model = self.df.loc[original_idx, self.model_column]
                    else:
                        original_mark = mark
                        original_model = model
                    
                    results.append(MatchResult(
                        mark=original_mark,
                        model=original_model,
                        mark_score=mark_score,
                        model_score=model_score,
                        combined_score=combined,
                        original_index=original_idx
                    ))
        
        # Sort by combined score and deduplicate
        results.sort(key=lambda x: -x.combined_score)
        
        # Deduplicate by (mark, model)
        seen = set()
        unique_results = []
        for r in results:
            key = (r.mark.lower(), r.model.lower())
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results[:top_k]


class HierarchicalMatcher(VehicleMatcher):
    """
    Hierarchical vehicle matcher with explicit two-step search:
    
    Step 1: Find top-N most similar brands
    Step 2: For each brand, find most similar models
    
    This approach is more structured and gives better control over matching.
    """
    
    def match_hierarchical(
        self,
        query_brand: str,
        query_model: str,
        top_n_brands: int = 3,
        top_n_models: int = 5,
        min_brand_score: float = 50.0,
        min_model_score: float = 40.0
    ) -> list[HierarchicalResult]:
        """
        Hierarchical matching: first brands, then models within each brand.
        
        Args:
            query_brand: User's brand query (can have typos, Russian or English).
            query_model: User's model query (can have typos).
            top_n_brands: Number of top brands to consider.
            top_n_models: Number of top models to return per brand.
            min_brand_score: Minimum score for brand matching (0-100).
            min_model_score: Minimum score for model matching (0-100).
            
        Returns:
            List of HierarchicalResult sorted by combined score.
        """
        results = []
        
        # Step 1: Find top-N similar brands
        similar_brands = self.find_similar_marks(
            query_brand, 
            top_k=top_n_brands, 
            min_score=min_brand_score
        )
        
        if not similar_brands:
            return []
        
        # Step 2: For each brand, find similar models
        for brand_rank, (brand, brand_score) in enumerate(similar_brands, 1):
            similar_models = self.find_similar_models(
                query_model,
                mark=brand,
                top_k=top_n_models,
                min_score=min_model_score
            )
            
            for model_rank, (model, model_score) in enumerate(similar_models, 1):
                # Combined score: weighted average
                combined = 0.4 * brand_score + 0.6 * model_score
                
                # Get original row index and names
                mask = (self.df['mark_clean'] == brand) & (self.df['model_clean'] == model)
                indices = self.df[mask].index.tolist()
                original_idx = indices[0] if indices else -1
                
                if original_idx >= 0:
                    original_brand = self.df.loc[original_idx, self.mark_column]
                    original_model = self.df.loc[original_idx, self.model_column]
                else:
                    original_brand = brand
                    original_model = model
                
                results.append(HierarchicalResult(
                    brand=original_brand,
                    model=original_model,
                    brand_score=brand_score,
                    model_score=model_score,
                    combined_score=combined,
                    rank_brand=brand_rank,
                    rank_model=model_rank,
                    original_index=original_idx
                ))
        
        # Sort by combined score
        results.sort(key=lambda x: -x.combined_score)
        return results
    
    def match_brands_with_models(
        self,
        query_brand: str,
        query_model: str,
        top_n_brands: int = 3,
        top_n_models: int = 5,
        min_brand_score: float = 50.0,
        min_model_score: float = 40.0
    ) -> list[BrandMatch]:
        """
        Get brands with their matching models (grouped by brand).
        
        Returns results grouped by brand, showing which models matched
        within each brand. Useful for displaying hierarchical results.
        
        Args:
            query_brand: User's brand query.
            query_model: User's model query.
            top_n_brands: Number of top brands to return.
            top_n_models: Number of top models per brand.
            min_brand_score: Minimum brand score.
            min_model_score: Minimum model score.
            
        Returns:
            List of BrandMatch objects, each containing brand and its models.
        """
        results = []
        
        # Step 1: Find similar brands
        similar_brands = self.find_similar_marks(
            query_brand,
            top_k=top_n_brands,
            min_score=min_brand_score
        )
        
        # Step 2: For each brand, find models
        for brand_clean, brand_score in similar_brands:
            similar_models = self.find_similar_models(
                query_model,
                mark=brand_clean,
                top_k=top_n_models,
                min_score=min_model_score
            )
            
            # Get original brand name
            if brand_clean in self.mark_to_indices:
                idx = self.mark_to_indices[brand_clean][0]
                original_brand = self.df.loc[idx, self.mark_column]
            else:
                original_brand = brand_clean
            
            # Get original model names
            models_with_original_names = []
            for model_clean, model_score in similar_models:
                mask = (self.df['mark_clean'] == brand_clean) & (self.df['model_clean'] == model_clean)
                indices = self.df[mask].index.tolist()
                if indices:
                    original_model = self.df.loc[indices[0], self.model_column]
                else:
                    original_model = model_clean
                models_with_original_names.append((original_model, model_score))
            
            results.append(BrandMatch(
                brand=original_brand,
                brand_score=brand_score,
                models=models_with_original_names
            ))
        
        return results
    
    def search(
        self,
        query_brand: str,
        query_model: str,
        top_n_brands: int = 3,
        top_n_models: int = 3,
        top_k: int = 5
    ) -> list[HierarchicalResult]:
        """
        Simple search interface - returns top-K results.
        
        This is a convenience wrapper around match_hierarchical.
        
        Args:
            query_brand: Brand query with possible typos.
            query_model: Model query with possible typos.
            top_n_brands: How many brands to consider.
            top_n_models: How many models per brand.
            top_k: Total number of results to return.
            
        Returns:
            Top-K HierarchicalResult sorted by combined score.
        """
        results = self.match_hierarchical(
            query_brand,
            query_model,
            top_n_brands=top_n_brands,
            top_n_models=top_n_models
        )
        return results[:top_k]


def find_similar_vehicles(
    parquet_path: str,
    query_mark: str,
    query_model: str,
    top_k: int = 5
) -> list[dict]:
    """
    Simple function to find similar vehicles from a parquet file.
    
    Args:
        parquet_path: Path to parquet file with 'mark' and 'model' columns.
        query_mark: User's mark query (can have typos, Russian or English).
        query_model: User's model query (can have typos).
        top_k: Number of results to return.
        
    Returns:
        List of dicts with 'mark', 'model', and 'score' keys.
        
    Example:
        >>> results = find_similar_vehicles(
        ...     'vehicles.parquet',
        ...     'мерседсе',  # typo in Mercedes
        ...     'е класс'    # E-Class in Russian
        ... )
        >>> print(results)
        [{'mark': 'Mercedes-Benz', 'model': 'E-Class', 'score': 92.5}, ...]
    """
    matcher = VehicleMatcher(parquet_path=parquet_path)
    results = matcher.match(query_mark, query_model, top_k=top_k)
    
    return [
        {
            'mark': r.mark,
            'model': r.model,
            'score': round(r.combined_score, 2),
            'mark_score': round(r.mark_score, 2),
            'model_score': round(r.model_score, 2)
        }
        for r in results
    ]

