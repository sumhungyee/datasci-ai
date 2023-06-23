
from ctransformers import AutoModelForCausalLM

from datasci_ai import model, AIFrame

import pandas as pd

path="C:\WizardCoder-15B-1.0.ggmlv3.q4_1.bin"
model_type="starcoder"
models = AutoModelForCausalLM.from_pretrained(path, model_type=model_type)



ddf = pd.DataFrame(data={'x': [i for i in range(10)], 'y': [2 * j + 3 for j in range(10)]})
aidf = AIFrame.AIDataFrame(model.LLM(llm_model=models), data=ddf)
print(aidf)
aidf.request("plot a line graph of y against x")

print(aidf)
newdf = aidf.request("create a new column of x * y")
print(newdf)



# tokens = model.tokenize(prompt)
# for token in model.generate(tokens):
#     print(model.detokenize(token), end="")


