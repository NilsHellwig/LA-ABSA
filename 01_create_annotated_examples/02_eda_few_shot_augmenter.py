import itertools
import random
import re, sys, os
import spacy
import nltk
from nltk.corpus import wordnet as wn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zero-shot-absa-quad/')))

from validator import validate_label

# Laden des spaCy-Modells
nlp = spacy.load("en_core_web_md")

nltk.download("wordnet")


def simple_tokenizer(text):
    return text.split(" ")


synonym_backup = {}


def get_synonym(word):
    if word in synonym_backup:
       if len(synonym_backup[word]) < 1:
            return word
       return random.choice(synonym_backup[word])
 
    doc = nlp(word)
    candidates = set()

    # Collect synonyms from WordNet
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():
                candidates.add(lemma_name)

    candidates = list([cand for cand in candidates if " " not in cand])
    
    synonym_backup[word] = candidates
    
    if len(candidates) < 1:
        return word

    if not candidates:
        return None

    # Rank candidates by similarity
    best_synonym = None
    best_score = -1

    for cand in candidates:
        cand_doc = nlp(cand)
        sim = doc.similarity(cand_doc)
        if sim > best_score:
            best_score = sim
            best_synonym = cand
            

    return best_synonym


def get_all_terms(tuples):
    return [t[0] for t in tuples if t[0] != "NULL"] + [
        t[3] for t in tuples if len(t) > 3 and t[3] != "NULL"
    ]


def augment_examples(file_path_save, task, dataset_name, n_few_shot):
    input_file = f"../zero-shot-absa-quad/fs_examples/{task}/{dataset_name}/fs_{n_few_shot}/examples.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()

    lines_save = []

    for idx, line in enumerate(lines):
        sentence_original = line.split("####")[0]
        tuples = eval(line.split("####")[1])
        sentence_tokens = simple_tokenizer(sentence_original)

        for k in range(int(2000 / len(lines))):
            
          if k%10 == 0:
              print(idx, ":", k)
          
          generation_valid = False
          n_gen = 0
          while generation_valid == False:
            tuples_aug = [list(t) for t in tuples.copy()]

            sentence_tokens_aug = sentence_tokens.copy()

            # 1. Random insertion
            random_word = random.choice(sentence_tokens_aug)
            random_position = random.randint(0, len(sentence_tokens_aug))
            sentence_tokens_aug.insert(random_position, get_synonym(random_word))

            # 2. random deletion of word that is non in get_all_terms(tuples_aug)
            random_word = random.choice(sentence_tokens_aug)
            sentence_tokens_aug.remove(random_word)

            # 3. Random Swap. This method takes two words in the sentence and swaps these words. Again we must make sure that the target words remain unchanged.
            random_word1 = random.choice(sentence_tokens_aug)
            random_word2 = random.choice(sentence_tokens_aug)
            index1 = sentence_tokens_aug.index(random_word1)
            index2 = sentence_tokens_aug.index(random_word2)
            sentence_tokens_aug[index1], sentence_tokens_aug[index2] = (
                        sentence_tokens_aug[index2],
                        sentence_tokens_aug[index1],
                    )

            # 4. Select random token in sentence_tokens_aug and replace it with a synonym. do also exchange the token in the tuples
            random_word = random.choice(sentence_tokens_aug)
            synonym = get_synonym(random_word)
            sentence_tokens_aug = [
                synonym if token == random_word else token
                for token in sentence_tokens_aug
            ]


            pattern = r'^\b{}\b'.format(re.escape(random_word))

            for t in tuples_aug:
               if bool(re.match(pattern, t[0])):
                  t[0] = re.sub(pattern, synonym, t[0], count=1)
               if len(t) > 3:
                  if bool(re.match(pattern, t[3])):
                      t[3] = re.sub(pattern, synonym, t[3], count=1)

            

            tuples_aug = [tuple(t) for t in tuples_aug]

            aug_sentence = " ".join(sentence_tokens_aug)
            if len(validate_label(tuples_aug, aug_sentence, is_string = False, check_unique_ac=False, unique_aspect_categories=[], task=task)) == 1:
                generation_valid = True
                lines_save.append(aug_sentence + "####" + str(tuples_aug))
                
                # print(aug_sentence)
                print(tuples_aug)
                
            else:
                n_gen += 1
                if n_gen > 10:
                    lines_save.append(sentence_original + "####" + str(tuples))
                    generation_valid = True
                    
                    

    with open(file_path_save, "w") as f:
        f.write("\n".join(lines_save))


n_few_shot = [10, 50]  # 0 fehlt noch
datasets = ["coursera", "rest16", "hotels", "flightabsa", "rest15"]
tasks = ["asqp", "tasd"]
TRANSLATE_TERMS = True

combinations = itertools.product(n_few_shot, datasets, tasks)

for combination in combinations:
    fs, dataset_name, task = combination
    file_path_save = f"_out_synthetic_examples/02_eda_few_shot_augmenter/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    augment_examples(
        file_path_save=file_path_save,
        task=task,
        dataset_name=dataset_name,
        n_few_shot=fs,
    )
