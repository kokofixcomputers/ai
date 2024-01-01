from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import os

# Define the model name
model_name = "mistralai/Mistral-7B-v0.1"

# Download the model to a specific path
cache_dir = "./your/cache/path"  # Specify your desired cache path
model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", cache_dir=cache_dir)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# The rest of the code for the chat interaction
conversation_history = [
    "What is the meaning of life?",
    "The meaning of life is to be happy."
]

def chat_with_bot(query):
    input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
    bot_response = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(bot_response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

user_query = ""
while user_query.lower() != "exit":
    user_query = input("You: ")
    if user_query.lower() != "exit":
        conversation_history.append(user_query)
        bot_reply = chat_with_bot(" ".join(conversation_history[-2:]))
        print("Bot:", bot_reply)
