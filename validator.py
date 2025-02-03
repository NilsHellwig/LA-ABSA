import re 

def extract_array_from_string(predicted_label):
    match = re.search(r"\[(.*\])", predicted_label)
    if match:
        return match.group(0)
    else:
        return None


def validate_label(predicted_label, input_text, unique_aspect_categories, polarities=["positive", "negative", "neutral"], task="asqp"):
        predicted_label = extract_array_from_string(predicted_label)
        if predicted_label == None:
            return [False, "no list in prediction"]
        
        try:
           label = eval(predicted_label)
        except:
           return [False, "not a list"]
        
        # 1. Check if the parsed object is a list
        try:
           if not isinstance(label, list):
              return [False, "not a list"]
        except:
           return [False, "not a list"]
        
        # 2. Check if the list contains exactly min one tuple
        if len(label) < 1:
            return [False, "no tuple found"]
        
        # 3. Check if the single element in the list is a tuple
        for element in label:
          if not isinstance(element, tuple):
             return [False, "inner elements not of type tuple"]

        
        # 4. Check if each element in the array is a tuple with exactly k elements
        n_elements_task = {"asqp": 4, "tasd": 3}
        for aspect in label:
          if len(aspect) != n_elements_task[task]:
              return [False, f"tuple has not exactly {n_elements_task[task]} elements"]
          
          for idx, item in enumerate(aspect):
            if not isinstance(item, str):
                return [False, "sentiment element not of type string"]
            
            # Check if sentiment element is empty string
            if len(item) < 1:
                return [False, "sentiment element string is empty string"]
            
            # Check if the 3rd value of the tuple is either 'positive', 'negative', or 'neutral'
            if item not in polarities and idx == 2:
                return [False, f"item {item} not a sentiment"]
            
            # Check if the category (2nd value) is in unique_aspect_categories
            if item not in unique_aspect_categories and idx == 1:
                return [False, f"item {item} is not a correct aspect category"]
            
        # 5. Check if terms are in sentence
        for _tuple in label:
            if not (_tuple[0] in input_text) and _tuple[0] != "NULL": 
                return [False, "aspect term not in text"]
            
            if task == "asqp":
               if not (_tuple[3] in input_text) and _tuple[3] != "NULL": 
                   return [False, "opinion term not in text"]
  
        
        # 6. If all checks pass, return the array
        return [label]
    
def validate_reasoning(output):
    if len(output) == 0:
        return [False, "no text found"]

    return [True]