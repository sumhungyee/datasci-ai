import pandas as pd
import time
import re
from ctransformers import AutoModelForCausalLM
from model import LLM, Reply
from tqdm import tqdm
from AIFrame import AIDataFrame