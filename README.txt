================================================================================
                         VEHICLE MATCHING PIPELINE
================================================================================

Fuzzy matching for vehicle marks and models with Russian/English support.
Handles typos, missing letters, transliteration, and language mixing.


INSTALLATION
--------------------------------------------------------------------------------

    pip install -r requirements.txt


QUICK START
--------------------------------------------------------------------------------

    from matcher import VehicleMatcher

    # Load your data (supports both 'brand' and 'mark' column names)
    matcher = VehicleMatcher(parquet_path='vehicles_db.parquet')

    # Search for similar vehicles
    results = matcher.match('мерседес', 'е класс', top_k=5)

    for r in results:
        print(f"{r.mark} {r.model} (score: {r.combined_score:.1f})")


PARQUET FILE FORMAT
--------------------------------------------------------------------------------

Your parquet file must have these columns:
  - brand or mark  : Vehicle brand/manufacturer (e.g., "Mercedes-Benz", "Toyota")
  - model          : Vehicle model (e.g., "E-Class", "Camry")


USAGE EXAMPLES
--------------------------------------------------------------------------------

1. Basic Search
   -------------
   from matcher import VehicleMatcher

   matcher = VehicleMatcher(parquet_path='vehicles_db.parquet')

   # Full search (mark + model)
   results = matcher.match('бмв', 'x5', top_k=5)


2. Search Only Brands
   -------------------
   marks = matcher.find_similar_marks('фолькваген', top_k=5)
   for mark, score in marks:
       print(f"{mark}: {score:.1f}")


3. Search Models Within a Brand
   ----------------------------
   models = matcher.find_similar_models('гольф', mark='volkswagen', top_k=5)
   for model, score in models:
       print(f"{model}: {score:.1f}")


4. Simple Function (One-off queries)
   ----------------------------------
   from matcher import find_similar_vehicles

   results = find_similar_vehicles(
       parquet_path='vehicles_db.parquet',
       query_mark='мерседсе',    # typo
       query_model='е класс',
       top_k=5
   )


FEATURES
--------------------------------------------------------------------------------

  * Fuzzy matching      - handles typos using multiple algorithms
                          (Levenshtein, Jaro-Winkler, token-based)
  
  * Transliteration     - Russian <-> English automatic conversion
  
  * Brand normalization - knows that "мерседес", "мерс", "mersedes" = "mercedes"
  
  * Auto column detect  - works with both 'brand' and 'mark' column names
  
  * Fast indexing       - pre-builds indexes for ~50k rows
  
  * Configurable        - adjust weights for mark vs model importance


MATCHING ALGORITHMS
--------------------------------------------------------------------------------

The pipeline combines multiple fuzzy matching strategies:

    Algorithm            Weight    Best For
    ------------------   ------    ---------------------------
    Levenshtein ratio    25%       Character-level typos
    Partial ratio        20%       Substring matching
    Token sort ratio     20%       Word order differences
    Token set ratio      15%       Extra/missing words
    Jaro-Winkler         20%       Prefix-preserving typos


EXAMPLE QUERIES
--------------------------------------------------------------------------------

    User Input                      Matches
    -----------------------------   ------------------------
    мерседес + е класс              Mercedes-Benz E-Class
    тоета + камри                   Toyota Camry
    бмв + x5                        BMW X5
    volkswagn + passat              Volkswagen Passat
    хёндай + солярис                Hyundai Solaris
    ауди + а4                       Audi A4
    форд + фокус                    Ford Focus


TESTING
--------------------------------------------------------------------------------

Open test_search.ipynb in Jupyter for interactive testing.


PROJECT STRUCTURE
--------------------------------------------------------------------------------

    VehiclesMatching/
    ├── matcher/
    │   ├── __init__.py          # Exports main functions
    │   ├── pipeline.py          # Main VehicleMatcher class
    │   ├── fuzzy.py             # Fuzzy matching algorithms
    │   └── transliteration.py   # Russian<->English conversion
    ├── vehicles_db.parquet      # Your vehicle data
    ├── test_search.ipynb        # Interactive testing notebook
    ├── example.py               # Usage examples
    ├── requirements.txt         # Dependencies
    ├── README.md                # This file (markdown)
    └── README.txt               # This file (plain text)

================================================================================

