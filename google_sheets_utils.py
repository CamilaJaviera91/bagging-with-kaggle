#Add libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

#Add warning and plt style
warnings.filterwarnings('ignore')
plt.style.use('default')

