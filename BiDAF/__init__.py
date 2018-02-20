
from hbconfig import Config
import numpy as np
import tensorflow as tf

from .attention import positional_encoding
from .encoder import Encoder
from .decoder import Decoder



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self):
        # 1. Contextual Embed Layer (Word Embedding, Charecter Embedding)
        # 2. Attention Flow Layer
        # 3. Modeling Layer
        # 4. Output Layer

        pass
