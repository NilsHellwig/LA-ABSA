import itertools
import random
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    """Holt eine Liste von Synonymen für ein gegebenes Wort."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

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

def synonym_replacement(sentence, targets):
    """Ersetzt zufällige Wörter im Satz durch Synonyme, außer den Zielbegriffen."""
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in targets]))
    random.shuffle(random_word_list)

    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

    return ' '.join(new_words)

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
                parts_copy += p.split()
        parts = parts_copy

        for _ in range(int(2000/ len(lines))):
            augmented_sentence = ""
            augmented_tuples = tuples.copy()
            
            # create list of length lines with random booleans. exactly 10% of the values are True
            n_true = int(len(parts) * 0.1)
            random_bools = [True] * n_true + [False] * (len(parts) - n_true)
            random.shuffle(random_bools)

            for i, part in enumerate(parts):
              if random_bools[i] == True:
                if part in terms and TRANSLATE_TERMS:
                    augmented_part = synonym_replacement(part, [part])
                    augmented_sentence += augmented_part + " "
                    augmented_tuples = [[augmented_part if t == part else t for t in tupl] for tupl in augmented_tuples]
                elif part in terms:
                    augmented_sentence += part + " "
                else:
                    augmented_part = synonym_replacement(part, terms)
                    augmented_sentence += augmented_part + " "
              else: 
                augmented_sentence += part + " "

            # save augmentations in format like this: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
            lines_save.append(f"{augmented_sentence.strip()}####{augmented_tuples}")

    with open(file_path_save, "w") as f:
        f.write("\n".join(lines_save))

n_few_shot = [10, 50]  # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["tasd", "asqp"]
TRANSLATE_TERMS = True

combinations = itertools.product(n_few_shot, datasets, tasks)

for combination in combinations:
    fs, dataset_name, task = combination
    file_path_save = f"augmentation/generations/eda/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    lines = augment_examples(file_path_save=file_path_save, task=task, dataset_name=dataset_name, n_few_shot=fs)
