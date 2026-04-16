import pandas as pd
import json
import re

def build_context(tx, users, sms_data, mail_data, location_data, iban_history):
    # tx is the suspicious transaction (Series)
    # users is the users.json data
    # location_data is locations.json data
    
    user_id = tx['sender_id']
    ts = pd.to_datetime(tx['timestamp'])
    
    # 1. User Profile
    user_profile = None
    # We need to find which user in users.json matches user_id (biotag)
    # The biotag matches the sender_id in transactions.
    # We can try to match by name or IBAN. 
    # Alain Regnier has IBAN FR85H4824371990132980420818.
    # In transactions, Alain is RGNR-LNAA-7FF-AUD-0.
    # Let's just find the user by IBAN if sender_iban is present.
    u_iban = tx['sender_iban']
    for u in users:
        if u['iban'] == u_iban:
            user_profile = u
            break
            
    # 2. Nearby Messages (SMS/Mail)
    nearby_messages = []
    # This part needs a way to link the user_id to the phone/email.
    # For now, let's just search for the user's first name in the messages.
    if user_profile:
        first_name = user_profile['first_name']
        
        # Check SMS
        for s in sms_data:
            content = s['sms']
            if first_name in content:
                # Extract date
                match = re.search(r'Date: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
                if match:
                    m_ts = pd.to_datetime(match.group(1))
                    diff = (ts - m_ts).total_seconds() / (24 * 3600)
                    if 0 <= diff <= 14:
                        nearby_messages.append(f"SMS ({diff:.1f} days ago): {content}")

        # Check Mails
        for m in mail_data:
            content = m['mail']
            if first_name in content:
                match = re.search(r'Date:.*(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2})', content)
                if match:
                    try:
                        m_ts = pd.to_datetime(match.group(1))
                        diff = (ts - m_ts).total_seconds() / (24 * 3600)
                        if 0 <= diff <= 14:
                            nearby_messages.append(f"Email ({diff:.1f} days ago): {content[:500]}...")
                    except:
                        pass

    # 3. Location info
    loc_info = "Unknown"
    user_locs = [l for l in location_data if l['biotag'] == user_id]
    if user_locs:
        # Find the closest location in time
        user_locs.sort(key=lambda x: abs((pd.to_datetime(x['timestamp']) - ts).total_seconds()))
        closest = user_locs[0]
        loc_info = f"{closest['city']} ({closest['lat']}, {closest['lng']}) at {closest['timestamp']}"

    # 4. IBAN History for recipient
    rid = tx['recipient_id']
    rhistory = list(iban_history.get(rid, []))

    context = {
        "transaction": tx.to_dict(),
        "user_profile": user_profile,
        "nearby_messages": nearby_messages,
        "location_info": loc_info,
        "recipient_iban_history": rhistory
    }
    
    return context
