from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Download the model to your local cache
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Run the model on a sample input
example = "My name is Wolfgang and I live in Berlin"
ner_results = nlp(example)
print(ner_results)
