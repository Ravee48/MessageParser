# New Code with optimised memory handling and response

from flask import Flask, request, jsonify
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

app = Flask(__name__)

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model_path = './Model/20250429_500.keras'
        model = load_model(model_path)
    
    if tokenizer is None:
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

def predict_messages_with_details(input_dict, max_len):
    load_model_and_tokenizer()  # Load model and tokenizer if not already loaded
    messages = [msg['message'] for msg in input_dict.get("msgs", [])]
    seqs = tokenizer.texts_to_sequences(messages)
    padded = pad_sequences(seqs, maxlen=max_len)
    
    predictions = model.predict(padded)
    results = {}
    
    for i, msg in enumerate(input_dict["msgs"]):
        message_id = msg['id']
        message = messages[i]
        is_transactional = predictions[i][0] > 0.35
        result = "Transactional" if is_transactional else "Non-Transactional"
        
        transaction_type = None
        amount = 0.0
        upi_id = None
        ref_no = None
        
        if is_transactional:
            credit_match = re.search(r'\b(credit|transfer|received|transferred|deposited|credited)\b', message, re.IGNORECASE)
            amount_matches = re.findall(r'(?<!\d\.\d{3}\.)(?:Rs?\.?\s*|INR\.?\s*|USD\.?\s*|debited by\s*|credited by\s*)([\d,]+(?:\.\d+)?)', message)    # re.findall(r'(?<!\d\.\d{3}\.)Rs?\.?\s*([\d,]+(?:\.\d{2})?)', message)
            upi_id_match = re.search(r'([\w.-]+)@([\w.-]+)', message, re.IGNORECASE)
            ref_no_match = re.findall(r'\b(?:Ref(?:erence)?(?:\s*No)?|RefNo|Ref#|Ref:)\.?\s*:?\s*(\d{12})', message, re.IGNORECASE)

            # Get currency type to check if it's USD
            currency_match = re.search(r'(USD)', message, re.IGNORECASE)
            # If message contains 'USD', convert to INR
            if amount_matches:
                # Remove commas and convert to float
                raw_amount = float(amount_matches[0].replace(',', ''))
                if currency_match:
                    amount = round(raw_amount * 86,2)  # Assuming USD to INR conversion
                else:
                    amount = raw_amount
            else:
                amount = 0.0

            # amount = amount_matches[0] if amount_matches else 0.0
            try:
                transaction_type = "Credit" if credit_match else "Debit"
                upi_id = upi_id_match[0] if upi_id_match else None
                ref_no = ref_no_match[0] if ref_no_match else None
            except Exception as e:
                print(f"Error extracting transaction_type, upi_id or ref_no: {e}")
                transaction_type = None
                upi_id = None
                ref_no = None
            
        results[message_id] = {
            "result": result,
            "transaction_type": transaction_type,
            "amount": amount,
            "prediction": float(predictions[i][0]),
            "upi_id": upi_id,
            "ref_no": ref_no
        }
    
    return results

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    max_len = 500  # Set your max_len here
    predictions = predict_messages_with_details(data, max_len)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    # app.run(debug=False)  # Set debug to False for production
