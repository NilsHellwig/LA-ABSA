import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

CACHE_FILE = "similarity_cache.json"

def load_cache():
    """Lädt die JSON-Datei mit gespeicherten Ähnlichkeitsscores."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Speichert die Ähnlichkeitsscores in die JSON-Datei."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)

def compute_similarity(text1, text2, model):
    """Berechnet die Kosinusähnlichkeit zwischen zwei Texten mithilfe von Sentence Transformers."""
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def sort_examples_by_similarity(example, examples, model_name="all-MiniLM-L6-v2"):
    """
    Sortiert die Liste `examples`, sodass das Beispiel mit dem ähnlichsten `text`-Wert zu `example["text"]` zuletzt steht.
    Verwendet einen Cache, um bereits berechnete Ähnlichkeiten zu speichern.

    :param example: Ein Dictionary mit dem Schlüssel "text".
    :param examples: Eine Liste von Dictionaries mit dem Schlüssel "text".
    :param model_name: Name des Transformer-Modells (Default: "all-MiniLM-L6-v2").
    :return: Sortierte Liste von Beispielen.
    """
    cache = load_cache()
    model = None  # Modell wird nur geladen, falls notwendig
    example_text = example["text"]

    similarities = []
    
    for ex in examples:
        example_pair_key = f"{example_text}###{ex['text']}"
        
        if example_pair_key in cache:
            similarity = cache[example_pair_key]
        else:
            # Modell nur laden, wenn eine neue Berechnung notwendig ist
            if model is None:
                model = SentenceTransformer(model_name)
            
            similarity = compute_similarity(example_text, ex["text"], model)
            cache[example_pair_key] = similarity
        
        similarities.append((similarity, ex))

    # Cache speichern, falls neue Berechnungen durchgeführt wurden
    save_cache(cache)

    # Sortiere die Beispiele nach Ähnlichkeit (ähnlichster zuletzt)
    sorted_examples = [ex for _, ex in sorted(similarities, key=lambda x: x[0])]

    return sorted_examples
