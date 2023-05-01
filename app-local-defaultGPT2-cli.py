from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, random_split

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(device)
model = model.to(torch.device(device))


text = "Data scientists use VS Code as their tool"

if device == 'cuda':
    input_ids = tokenizer.encode(text, return_tensors='pt').cuda()
else:
    input_ids = tokenizer.encode(text, return_tensors='pt')


output = model.generate(input_ids, max_length=100, do_sample=True)


print(tokenizer.decode(output[0], skip_special_tokens=True))



