"""
Example usage of the vehicle matching pipeline.
"""

import pandas as pd
from matcher import VehicleMatcher, find_similar_vehicles


def main():
    # === Option 1: Simple function call ===
    # If you just want quick results:
    
    results = find_similar_vehicles(
        parquet_path='vehicles.parquet',
        query_mark='мерседсе',      # typo in "мерседес" (Mercedes)
        query_model='е класс',       # "E-Class" in Russian
        top_k=5
    )
    
    print("=== Simple function results ===")
    for r in results:
        print(f"  {r['mark']} {r['model']} (score: {r['score']})")
    
    
    # === Option 2: Using the VehicleMatcher class ===
    # Better for multiple queries (loads data once)
    
    matcher = VehicleMatcher(parquet_path='vehicles.parquet')
    
    # Test queries with typos and Russian/English mix
    test_queries = [
        ('мерседес', 'e class'),        # Russian mark, English model
        ('mersedes', 'е класс'),         # Transliterated mark with typo
        ('тоета', 'камри'),              # Toyota Camry in Russian with typo
        ('toyot', 'camr'),               # Truncated words
        ('бмв', 'x5'),                   # BMW X5 in Russian
        ('volkswagn', 'passat'),         # Typo in Volkswagen
        ('хёндай', 'солярис'),           # Hyundai Solaris in Russian
        ('hyunday', 'solaris'),          # Typo in Hyundai
    ]
    
    print("\n=== Matcher class results ===")
    for query_mark, query_model in test_queries:
        results = matcher.match(query_mark, query_model, top_k=3)
        
        print(f"\nQuery: '{query_mark}' + '{query_model}'")
        if results:
            for r in results:
                print(f"  → {r.mark} {r.model} (combined: {r.combined_score:.1f}, "
                      f"mark: {r.mark_score:.1f}, model: {r.model_score:.1f})")
        else:
            print("  No matches found")
    
    
    # === Option 3: Search only marks or only models ===
    
    print("\n=== Mark search only ===")
    marks = matcher.find_similar_marks('фолькваген', top_k=5)  # Volkswagen with typo
    for mark, score in marks:
        print(f"  {mark}: {score:.1f}")
    
    print("\n=== Model search (filtered by mark) ===")
    models = matcher.find_similar_models('гольф', mark='volkswagen', top_k=5)
    for model, score in models:
        print(f"  {model}: {score:.1f}")


def create_sample_data():
    """Create sample vehicles.parquet for testing."""
    
    data = {
        'mark': [
            'Mercedes-Benz', 'Mercedes-Benz', 'Mercedes-Benz', 'Mercedes-Benz',
            'BMW', 'BMW', 'BMW', 'BMW',
            'Audi', 'Audi', 'Audi',
            'Volkswagen', 'Volkswagen', 'Volkswagen', 'Volkswagen',
            'Toyota', 'Toyota', 'Toyota', 'Toyota',
            'Honda', 'Honda', 'Honda',
            'Hyundai', 'Hyundai', 'Hyundai',
            'Kia', 'Kia', 'Kia',
            'Nissan', 'Nissan', 'Nissan',
            'Mazda', 'Mazda',
            'Ford', 'Ford', 'Ford',
            'Chevrolet', 'Chevrolet',
            'LADA', 'LADA', 'LADA',
        ],
        'model': [
            'E-Class', 'C-Class', 'S-Class', 'GLE',
            'X5', 'X3', '5 Series', '3 Series',
            'A4', 'A6', 'Q5',
            'Passat', 'Golf', 'Tiguan', 'Polo',
            'Camry', 'Corolla', 'RAV4', 'Land Cruiser',
            'Civic', 'Accord', 'CR-V',
            'Solaris', 'Tucson', 'Santa Fe',
            'Rio', 'Sportage', 'Ceed',
            'Qashqai', 'X-Trail', 'Almera',
            'CX-5', '6',
            'Focus', 'Mondeo', 'Kuga',
            'Cruze', 'Captiva',
            'Vesta', 'Granta', 'XRAY',
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_parquet('vehicles.parquet', index=False)
    print(f"Created sample vehicles.parquet with {len(df)} rows")
    return df


if __name__ == '__main__':
    import os
    
    # Create sample data if not exists
    if not os.path.exists('vehicles.parquet'):
        print("Creating sample data...")
        create_sample_data()
        print()
    
    main()

