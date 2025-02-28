import itertools
import nlpaug.augmenter.word as naw
import torch


def split_sentence(sentence, targets):
    """Teilt den Satz so auf, dass die Zielbegriffe optional übersetzt werden können."""
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


cache = {}

def back_translate(text, candidate_langs):
    """Führt die Backtranslation mit mehreren Sprachen durch und verwendet ein Cache für häufige Übersetzungen."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Backtranslation Augmenter für jede Sprache
    all_back_translation_aug = [
        naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-en-{lang}',
            to_model_name=f'Helsinki-NLP/opus-mt-{lang}-en',
            device=device
        ) for lang in candidate_langs
    ]
    
    
    augmented_texts = []
    
    for aug, lang in zip(all_back_translation_aug, candidate_langs):
        augmented_text = []
        for word in text.split():
            # Wenn der Begriff bereits im Cache ist, verwende die zwischengespeicherte Übersetzung
            if (word, lang) in cache:
                augmented_text.append(cache[(word, lang)])
            else:
                # Andernfalls übersetze den Begriff
                translated_word = aug.augment(word)
                if len(translated_word) > 0:
                    augmented_word = translated_word[0]
                    augmented_text.append(augmented_word)
                    # Cache die Übersetzung für zukünftige Anfragen
                    cache[(word, lang)] = augmented_word
                else:
                    augmented_text.append(word)
        
        augmented_texts.append(" ".join(augmented_text))
    augmented_texts = [txt.replace("\n", " ").replace(".", " ").replace("?", " ").replace("!", " ") for txt in augmented_texts]
    augmented_texts = [txt if len(text) * 2 > len(txt) else text for txt in augmented_texts ] # remove rare cases were extreme long texts are generated
    return augmented_texts

def augment_examples(file_path_save, task, dataset_name, n_few_shot):
    input_file = f"./fs_examples/{task}/{dataset_name}/fs_{n_few_shot}/examples.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()
        
    lines_save = []
    
    for idx, line in enumerate(lines):
        print(idx,":", line)
        # Example line: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
        sentence_original = line.split("####")[0]
        tuples_raw = line.split("####")[1]
        tuples = eval(tuples_raw)

        # get all aspect terms (index 0) and opinion terms (if index 3 exists)
        aspect_terms = [t[0] for t in tuples if t[0] != "NULL"]
        opinion_terms = [t[3] for t in tuples if len(t) > 3 and t[3] != "NULL"]
        # merge terms into one list
        terms = aspect_terms + opinion_terms
        
        # split sentece by terms go character by character
        parts = split_sentence(sentence_original, terms)
        
        augmented_sentence = [""] * len(candidate_langs)
        augmented_tuples = [tuples for _ in candidate_langs]
        
        
        for part in parts:
          part_backtranslated = back_translate(part, candidate_langs)
          for i, lang in enumerate(candidate_langs):
            if part in terms:
                if TRANSLATE_TERMS:
                    augmented_sentence[i] += part_backtranslated[i] + " "
                    augmented_tuples[i] = [[part_backtranslated[i] if t==part else t for t in tupl] for tupl in augmented_tuples[i]]
                else:
                    augmented_sentence[i] += part + " "
                    augmented_tuples[i] = [[t for t in tupl] for tupl in augmented_tuples[i]]
                
            else:
                augmented_sentence[i] += part_backtranslated[i] + " "
                augmented_tuples[i] = [[part_backtranslated[i] if t==part else t for t in tupl] for tupl in augmented_tuples[i]]
        
        # update aspect terms and opinion terms in tuples with backtranslated terms
        # print("Sentence:", augmented_sentence[0], "####", sentence_original)
        # print("Tuples:", augmented_tuples[0])
        
        # save augmentations in format like this: Much of the time it seems like they do not care about you .####[['NULL', 'service general', 'negative', 'do not care about you']]
        for i, lang in enumerate(candidate_langs):
            lines_save.append(f"{augmented_sentence[i].strip()}####{augmented_tuples[i]}")
        
    with open(file_path_save, "w") as f:
        f.write("\n".join(lines_save))
        
        
                
        
n_few_shot = [10, 50] # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["tasd", "asqp"]
TRANSLATE_TERMS = True
candidate_langs = ['fr', 'zh', 'ar', 'jap', 'da']


combinations = itertools.product(n_few_shot, datasets, tasks)       
        

for combination in combinations:
    fs,  dataset_name, task = combination
    file_path_save = f"_out_synthetic_examples/04_back_translation_few_shot_augmenter/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    lines = augment_examples(file_path_save=file_path_save, task=task, dataset_name=dataset_name, n_few_shot=fs)
    
