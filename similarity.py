import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

CACHE_FILE = "./01_create_annotated_examples/similarity_cache.json"

def load_cache():
    """Lädt die JSON-Datei mit gespeicherten Ähnlichkeitsscores, falls sie gültig ist."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Warnung: similarity_cache.json ist ungültig oder beschädigt. Erstelle neuen Cache.")
            return {}
    return {}


def save_cache(cache):
    """Speichert die Ähnlichkeitsscores in die JSON-Datei."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        # Konvertiere alle numpy.float32-Werte in normale float-Werte
        cache = {key: float(value) for key, value in cache.items()}
        json.dump(cache, f, indent=4, ensure_ascii=False)


def compute_similarity(text1, text2, model):
    """Berechnet die Kosinusähnlichkeit zwischen zwei Texten."""
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)  # Hier explizit in float umwandeln


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
