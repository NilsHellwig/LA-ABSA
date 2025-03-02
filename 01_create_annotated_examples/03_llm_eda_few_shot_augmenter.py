import itertools
import random
import re
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import LLM

synonym_cache = {}

llm = LLM("llama3.1:8b", parameters=[
        {"name": "stop", "value": [")]"]}, 
        {"name": "num_ctx", "value": "4096"}
        ]) #8192


def extract_text_between_quotes(text):
  return re.findall(r'"([^"]*)"', text)

def get_prompt(word, sentence_original):
    return f'''I'll give you a term that is in the sentence "{sentence_original}", find a similar term and return it enclosed by '"'.\nTerm: "{word}"\nSimilar Term: "'''
    

def get_synonyms(word, sentence_original, top_n=5):
    """Holt eine Liste von Synonymen für ein gegebenes Wort basierend auf Wortähnlichkeiten."""
    # Überprüfen, ob das Wort bereits im Cache ist
    if word + "_" + sentence_original in synonym_cache:
        return synonym_cache[word + "_" + sentence_original]

    synonyms = []
    if synonym_cache.get(word + "_" + sentence_original):
        synonyms = synonym_cache.get(word + "_" + sentence_original)
        return synonyms
    
    seed = 0
    
    while len(synonyms) < top_n:
        output, duration = llm.predict(get_prompt(word, sentence_original), seed=seed)
        extraction = extract_text_between_quotes(output)
        if len(extraction) > 0 and len(extraction[0]) > 0 and extraction[0] not in synonyms and "imilar word" not in extraction[0] and extraction[0] != word and word != "" and word != " ":
            synonyms.append(extraction[0])
        if seed > 20:
            synonyms.append(word)
        seed += 1
    print(word, ":",synonyms)
    
    synonym_cache[word + "_" + sentence_original] = synonyms

    return synonyms

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

def synonym_replacement(target, n_aug, sentence_original):
    """Ersetzt zufällige Wörter im Satz durch Synonyme, außer den Zielbegriffen."""
    words = simple_tokenizer(target)
    words_bool = [True] * len(words)
    random.shuffle(words_bool)
    words_with_syn_repalcement = []
    for word, bool_ in zip(words, words_bool):
        if bool_ and word.isalpha():
            synonyms = get_synonyms(word, sentence_original)
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
        print(idx)
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
                    augmented_part = synonym_replacement(part, n_aug, sentence_original)
                    augmented_sentence += augmented_part + " "
                # Normaler Term und n_aug == 0
                elif part not in terms and n_aug == 0:
                    augmented_sentence += part + " "
                # Term und n_aug > 0
                elif part in terms and n_aug > 0 and TRANSLATE_TERMS:
                    augmented_part = synonym_replacement(part, n_aug, sentence_original)
                    augmented_sentence += augmented_part + " "
                    augmented_tuples = [[augmented_part if t == part else t for t in tupl] for tupl in augmented_tuples]
                # Term und n_aug == 0
                else:
                    augmented_sentence += part + " "
                
                current_token += len(simple_tokenizer(part)) 
                
            # save augmentations in format like this: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
            lines_save.append(f"{augmented_sentence.strip()}####{augmented_tuples}")

    with open(file_path_save, "w") as f:
        f.write("\n".join(lines_save))

n_few_shot = [10, 50]  # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["asqp", "tasd"]
TRANSLATE_TERMS = True

combinations = itertools.product(n_few_shot, datasets, tasks)

for combination in combinations:
    fs, dataset_name, task = combination
    file_path_save = f"_out_synthetic_examples/03_llm_eda_few_shot_augmenter/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    lines = augment_examples(file_path_save=file_path_save, task=task, dataset_name=dataset_name, n_few_shot=fs)
