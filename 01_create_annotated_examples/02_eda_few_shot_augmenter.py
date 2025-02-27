import itertools
import random
import re
import spacy

# Laden des spaCy-Modells
nlp = spacy.load("en_core_web_md")

synonym_cache = {}


def get_synonyms(word, top_n=5):
    """Holt eine Liste von Synonymen für ein gegebenes Wort basierend auf Wortähnlichkeiten."""
    # Überprüfen, ob das Wort bereits im Cache ist
    if word in synonym_cache:
        return synonym_cache[word]

    # Wort mit spaCy verarbeiten
    word_doc = nlp(word)

    # Synonyme basierend auf Wortähnlichkeiten finden
    synonyms = [w.text for w in word_doc.vocab if w.is_lower and w.has_vector and w.similarity(word_doc) > 0.5]
    synonyms = sorted(synonyms, key=lambda w: word_doc.similarity(nlp(w)), reverse=True)

    # Die Top-N Synonyme speichern
    top_synonyms = synonyms[:top_n]

    # Ergebnis im Cache speichern
    synonym_cache[word] = top_synonyms

    return top_synonyms

def split_sentence(sentence, targets):
    """Teilt den Satz so auf, dass die Zielbegriffe optional synonymisiert werden können."""
    parts = []
    current_part = ""
    i = 0
    while i < len(sentence):
        matched = False
        for target in targets:
            if sentence[i:i+len(target)] == target:
                if current_part:
                    parts.append(current_part.strip())
                parts.append(target)
                i += len(target)
                current_part = ""
                matched = True
                break
        if not matched:
            current_part += sentence[i]
            i += 1
    if current_part:
        parts.append(current_part.strip())
    return parts

def synonym_replacement(target, n_aug):
    """Ersetzt zufällige Wörter im Satz durch Synonyme, außer den Zielbegriffen."""
    words = simple_tokenizer(target)
    words_bool = [True] * len(words)
    random.shuffle(words_bool)
    words_with_syn_repalcement = []
    for word, bool_ in zip(words, words_bool):
        if bool_ and word.isalpha():
            synonyms = get_synonyms(word)
            if synonyms:
                words_with_syn_repalcement.append(random.choice(synonyms))
            else:
                words_with_syn_repalcement.append(word)
        else:
            words_with_syn_repalcement.append(word)

    return " ".join(words_with_syn_repalcement)

def simple_tokenizer(text):
    # Verwenden Sie reguläre Ausdrücke, um Wörter und Satzzeichen zu trennen
    tokens = re.findall(r'\b\w+\b|[.,!?;:]', text)
    return tokens

def augment_examples(file_path_save, task, dataset_name, n_few_shot):
    input_file = f"./fs_examples/{task}/{dataset_name}/fs_{n_few_shot}/examples.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()

    lines_save = []

    for idx, line in enumerate(lines):
        print(idx, ":", line)
        # Example line: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
        sentence_original = line.split("####")[0]
        tuples_raw = line.split("####")[1]
        tuples = eval(tuples_raw)

        # get all aspect terms (index 0) and opinion terms (if index 3 exists)
        aspect_terms = [t[0] for t in tuples if t[0] != "NULL"]
        opinion_terms = [t[3] for t in tuples if len(t) > 3 and t[3] != "NULL"]
        # merge terms into one list
        terms = aspect_terms + opinion_terms

        # split sentence by terms go character by character
        parts = split_sentence(sentence_original, terms)
        
        parts_copy = []
        
        for p in parts:
            if p in terms:
                parts_copy.append(p)
            else:
                parts_copy += simple_tokenizer(p)
        parts = parts_copy

        for _ in range(int(2000/ len(lines))):
            augmented_sentence = ""
            augmented_tuples = tuples.copy()
            
            # create list of length lines with random booleans. exactly 10% of the values are True
            n_true = int(len(simple_tokenizer(sentence_original)) * 0.1)
            if n_true == 0:
                n_true = 1
            
            random_bools = [True] * n_true + [False] * (len(parts) - n_true)
            random.shuffle(random_bools)
            
            current_token = 0

            for i, part in enumerate(parts):
                n_aug = random_bools[current_token:current_token + len(simple_tokenizer(part))].count(True)
                # Normaler Term und n_aug > 0
                if part not in terms and n_aug > 0:
                    augmented_part = synonym_replacement(part, n_aug)
                    augmented_sentence += augmented_part + " "
                # Normaler Term und n_aug == 0
                elif part not in terms and n_aug == 0:
                    augmented_sentence += part + " "
                # Term und n_aug > 0
                elif part in terms and n_aug > 0 and TRANSLATE_TERMS:
                    augmented_part = synonym_replacement(part, n_aug)
                    augmented_sentence += augmented_part + " "
                    augmented_tuples = [[augmented_part if t == part else t for t in tupl] for tupl in augmented_tuples]
                # Term und n_aug == 0
                elif n_aug == 0:
                    augmented_sentence += part + " "
                
                current_token += len(simple_tokenizer(part)) 
                
            # save augmentations in format like this: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
            lines_save.append(f"{augmented_sentence.strip()}####{augmented_tuples}")

    with open(file_path_save, "w") as f:
        f.write("\n".join(lines_save))

n_few_shot = [10, 50]  # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["tasd", "asqp"]
TRANSLATE_TERMS = False

combinations = itertools.product(n_few_shot, datasets, tasks)

for combination in combinations:
    fs, dataset_name, task = combination
    file_path_save = f"augmentation/generations/eda/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    lines = augment_examples(file_path_save=file_path_save, task=task, dataset_name=dataset_name, n_few_shot=fs)
