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

#current directory
FOLDER = './'

#spreadsheet ID (you can find it in the URL of your Google Sheet example: https://docs.google.com/spreadsheets/d/1aBcD1234EfGhI56789JKLMnOpQrStUvWxYz/edit#gid=0)
#"1aBcD1234EfGhI56789JKLMnOpQrStUvWxYz" this is the id that we need
SPREADSHEET_ID = 'vsfGazjkHf8x2soh53kJJzuUYqyuA_ZyiuZU'

#path to the credentials file
#the credentials looks like this:
#{
#   "type": "",
#   "project_id": "",
#   "private_key_id": "",
#   "private_key": "",
#   "client_email": "",
#   "client_id": "",
#   "auth_uri": "",
#   "token_uri": "",
#   "auth_provider_x509_cert_url": "",
#   "client_x509_cert_url": "",
#   "universe_domain": ""
# }
SERVICE_ACCOUNT_FILE = FOLDER + 'credentials.json'

#scopes required to access googlesheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 
          'https://spreadsheets.google.com/feeds', 
          'https://www.googleapis.com/auth/drive.file',
          'https://www.googleapis.com/auth/drive']

#load the credentials from the JSON file
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)