from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load model and tokenizer from "model" directory
model_name = "sebasatarama/F-DetectorModel"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive the JSON
    sentences = data['sentences']  # Extract sentences

    predictions = []
    for sentence in sentences:
        # Preprocess and tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted probabilities using softmax
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
        
        # Find the maximum probability and corresponding label
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()  # Convert tensor to a float value
        predicted_class = predicted_class.item()  # Convert tensor to int
        
        # Check if the max probability is greater than 70%
        if max_prob > 0.7:
            # Return the predicted label and probability
            predictions.append({
                "sentence": sentence,
                "label": predicted_class,
                "probability": round(max_prob * 100, 2)  # Return as a percentage
            })

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
