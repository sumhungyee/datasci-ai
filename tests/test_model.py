

from datasci_ai import model
import pandas as pd
import numpy as np
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.llm = model.LLM(path="C:\WizardCoder-15B-1.0.ggmlv3.q4_1.bin")
        self.df = pd.DataFrame(data={'x': [i for i in range(10)], 'y': [2 * j + 3 for j in range(10)]})

        

    def test_generate_tokens(self):
        prompt = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request in markdown format, using one code block without explanations.
        
        ### Instruction: write a hello world statement in java
        ### Response:
        """
        tokens = self.llm.model.tokenize(prompt)

        reply=""
        for token in self.llm.model.generate(tokens):
            next_token = self.llm.model.detokenize(token)
            print(next_token, end="")
            reply+=next_token

        assert(bool(reply))

    def test_generate_pandas(self):
        
        prompt = """
        Below is an instruction that describes a programming task. Write a response in python that appropriately completes the request in markdown format, using one code block without explanations.
        
        ### Instruction: You are provided with a pandas dataframe. This dataframe has {self.df.shape[0]} rows and {self.df.shape[1]} columns, and the list of its columns are {self.df.columns}. Plot a straight line graph of y and x.
        ### Response:
        """
        tokens = self.llm.model.tokenize(prompt)

        reply=""
        for token in self.llm.model.generate(tokens):
            next_token = self.llm.model.detokenize(token)
            print(next_token, end="")
            reply+=next_token

        assert(bool(reply))

    

            

if __name__ == '__main__':
    unittest.main()