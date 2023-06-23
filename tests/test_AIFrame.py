from datasci_ai import model, errors, AIFrame
import pandas as pd
import numpy as np
import re
import unittest


class AIFrameTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(data={'x': [i for i in range(10)], 'y': [2 * j + 3 for j in range(10)]})
        self.llm = model.LLM(path="C:\WizardCoder-15B-1.0.ggmlv3.q4_1.bin")
        self.aidf = AIFrame.AIDataFrame(llm=self.llm)

    def test_query_1(self):
        self.aidf.request("Plot a line plot of y against x.", verbose=True)
    
    def test_query_2(self):
        newdf = self.aidf.request("create a 3rd column z with values of x * y. Then, plot a graph of z against y", verbose=True)
        print(newdf)
