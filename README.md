# datasci-ai
Inspired by pandasai, datasci-ai strives to localise LLMs for data analysis.

## Installation

```
pip install git+https://github.com/sumhungyee/datasci-ai.git
```

## Usage
1. Perform imports
```python
from datasci_ai import LLM, AIDataFrame
```
1. Load LLM (Currently only supports WizardCoder 15B ggml)
```python
llm = LLM() # Defaults to downloading WizardCoder-15B-1.0.ggmlv3.q4_1.bin off Huggingface
```
```python
llm = LLM(path="C:\WizardCoder-15B-1.0.ggmlv3.q5_0.bin") # To a path
```
```python
llm = LLM(temperature = 0.5, model_file='WizardCoder-15B-1.0.ggmlv3.q4_1.bin', link='TheBloke/WizardCoder-15B-1.0-GGML') # Note that only WizardCoder 15B ggml models are support at the moment
```
1. Load data
```python
df = AIDataFrame(llm, data=pd.read_csv(...))
```
1. Instruct LLM
```python
new_df = df.instruct("Create a new column called 'score_groups', splitting the score column into 5 intervals. The numbers for score range from 0 to 10")
# Returns updated AIDataFrame
```
```python
df.instruct("Plot a histogram of 'Popularity' for each score group in 'score_groups', with one histogram per figure", verbose=False) # Mutes generated code.
```

## Future plans
1. Rework README.md
2. Allow GPU offloading
3. Possible GPTQ quantisation
