import json
import pandas as pd
import os

def load_transactions(file_path):
    return pd.read_csv(file_path)

def load_users(file_path):
    with open(file_path, 'r') as f:
        users = json.load(f)
    return {user['iban']: user for user in users} # Keyed by IBAN if we need quick lookup from transaction

def load_users_by_biotag(file_path):
    with open(file_path, 'r') as f:
        users = json.load(f)
    # Extract biotag from user data? The briefing mentions biotags like RGNR-LNAA-7FF-AUD-0.
    # Looking at train/users.json, there are no biotags in the objects... 
    # Wait, the briefing says: "The three users and their IBAN anchors:"
    # RGNR-LNAA-7FF-AUD-0 | Alain Regnier | ...
    # I should check users.json again to see if biotag is there.
    return users

def load_sms(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_mails(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_locations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
