import sys
from time import time
import os


tools_path = f"{os.path.dirname(__file__)}/tools_mk"
sys.path.append(tools_path)

from emails_Process_mk import preprocess_mk

features_train_transformed, features_test_transformed, labels_train, labels_test = preprocess_mk()





