# app.py
#
# Use this sample code to handle webhook events in your integration.
#
# 1) Paste this code into a new file (app.py)
#
# 2) Install dependencies
#   pip3 install flask
#   pip3 install stripe
#
# 3) Run the server on http://localhost:4242
#   python3 -m flask run --port=4242

import stripe
import yaml
from flask import Flask, jsonify, request

# The library needs to be configured with your account's secret key.
# Ensure the key is kept out of any version control system you might be using.
stripe.api_key = "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"

# This is your Stripe CLI webhook secret for testing your endpoint locally.
endpoint_secret = 'whsec_9dc15ffce726dbb509d25f407418a6c849b66cf7e01e46c4c678b54a10ff4db3'

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    event = None
    payload = request.data
    sig_header = request.headers['STRIPE_SIGNATURE']

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        raise e
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        raise e

    if event['type'] == 'invoice.payment_succeeded':
        session = event['data']['object']
        # Fulfill the purchase...
        print("-----------------------------------------------------------------------------")
        print("invoice.payment_succeeded")
        print("session: ", session)
        print("-----------------------------------------------------------------------------")
        # Apri il file yaml in modalità lettura
        with open('config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # Modifica i dati
        data['preauthorized']['emails'].append(session.customer_email)

        # Apri il file yaml in modalità scrittura
        with open('config.yaml', 'w') as file:
            # Scrivi i dati modificati nel file
            yaml.safe_dump(data, file)


    return jsonify(success=True)

    

if __name__ == '__main__':
    app.run(port=4242)