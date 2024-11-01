import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load a pre-trained BERT model for token classification (NER as an example)
model_name = "dslim/bert-base-NER"  # This is a BERT model fine-tuned for NER (Named Entity Recognition)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the pipeline for token classification (like NER)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define the function for slot filling
def fill_slots(sentence):
    # Use the pipeline to get token classifications (NER-style)
    entities = nlp(sentence)
    print(entities)
    # Initialize an empty dictionary to hold the slots
    slots = {"action": "", "id": "", "location": "", "target": ""}
    for entity in entities:
        print(entity)
    # A simple rule-based approach to assign entities to slots
    for entity in entities:
        if entity["entity_group"] == "PER":  # You can customize based on your data
            slots["target"] = entity["word"]
        elif entity["entity_group"] == "ORG" or entity["entity_group"] == "MISC":
            slots["id"] = entity["word"]
        elif entity["entity_group"] == "LOC":
            slots["location"] = entity["word"]
    
    # A heuristic to extract the "action" from the sentence
    tokens = tokenizer.tokenize(sentence)
    if "Bring" in tokens or "take" in tokens:
        slots["action"] = "bring"  # You can make this more dynamic with custom rules
    
    return slots

# Example sentence
s1 = "Bring me the coke from the dinner table"

# Get the slot-filled result
slots = fill_slots(s1)
print(slots)
