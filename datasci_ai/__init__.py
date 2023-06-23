import pandas as pd
import time
import re
from ctransformers import AutoModelForCausalLM
from datasci_ai.AIFrame import AIDataFrame
from datasci_ai.model import LLM, Reply
from tqdm import tqdm
