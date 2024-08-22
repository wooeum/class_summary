import pickle
import chromadb
import pandas as pd
import numpy as np
import cupy as cp

from pathlib import Path
from PIL import Image

client = chromadb.PersistentClient(path='/content/drive/MyDrive/chromadb')