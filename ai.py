from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Download the model to your local cache
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForTokenClassification.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Run the model on a sample input
example = "My name is Wolfgang and I live in Berlin"
ner_results = nlp(example)
print(ner_results)