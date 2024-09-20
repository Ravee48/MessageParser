# New Code with optimised memory handling

from flask import Flask, request, jsonify
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

app = Flask(__name__)

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model_path = './Model/best_model_2_500_NoiseNotRemoved'
        model = load_model(model_path)
    
    if tokenizer is None:
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

def predict_messages_with_details(input_dict, max_len):
    load_model_and_tokenizer()  # Load model and tokenizer if not already loaded
    messages = input_dict.get("msgs", [])
    seqs = tokenizer.texts_to_sequences(messages)
    padded = pad_sequences(seqs, maxlen=max_len)
    
    predictions = model.predict(padded)
    results = {}
    
    for i, message in enumerate(messages):
        is_transactional = predictions[i][0] > 0.35
        result = "Transactional" if is_transactional else "Non-Transactional"
        
        transaction_type = None
        amount = None
        
        if is_transactional:
            credit_match = re.search(r'\b(credit|transfer|received|transferred|credited)\b', message, re.IGNORECASE)
            amount_matches = re.findall(r'(?<!\d\.\d{3}\.)Rs?\.?\s*([\d,]+(?:\.\d{2})?)', message)
            amount = amount_matches[0] if amount_matches else None
            
            transaction_type = "Credit" if credit_match else "Debit"

        results[message] = {
            "result": result,
            "transaction_type": transaction_type,
            "amount": amount,
            "prediction": float(predictions[i][0])
        }
    
    return results

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    max_len = 500  # Set your max_len here
    predictions = predict_messages_with_details(data, max_len)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=False)  # Set debug to False for production


# OLD Code

"""
from flask import Flask, request, jsonify
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle


app = Flask(__name__)


tokenizer = Tokenizer(num_words=10000)
# Load your model here
model_path = './Model/best_model_2_500_NoiseNotRemoved'  # Update with your model path
model = load_model(model_path)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# # Load the tokenizer
# with open('tokenizer.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)

def predict_messages_with_details(model, input_dict, max_len):
    max_len = max_len
    predictions = {}
    messages = input_dict.get("msgs", [])
    
    for message in messages:
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=max_len)
        prediction = model.predict(padded)
        
        # Determine if the message is transactional
        is_transactional = prediction[0][0] > 0.35
        result = "Transactional" if is_transactional else "Non-Transactional"
        
        # Initialize details
        transaction_type = None
        amount = None
        
        if is_transactional:
            credit_match = re.search(r'\b(credit|transfer|received|transferred|credited)\b', message, re.IGNORECASE)
            amount_matches = re.findall(r'(?<!\d\.\d{3}\.)Rs?\.?\s*([\d,]+(?:\.\d{2})?)', message)
            amount = amount_matches[0] if amount_matches else None
            
            if credit_match:
                transaction_type = "Credit"
            else:
                transaction_type = "Debit"

        # Store the results
        predictions[message] = {
            "result": result,
            "transaction_type": transaction_type,
            "amount": amount,
            "prediction": float(prediction[0][0])
        }
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    max_len = 500  # Set your max_len here
    predictions = predict_messages_with_details(model, data, max_len)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=True)
"""
