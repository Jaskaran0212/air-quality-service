from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle

class AirQualityLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=100, output_size=13):
        super(AirQualityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

app = Flask(__name__)

# Load model and scaler
model = AirQualityLSTM()
model.load_state_dict(torch.load('model/lstm_model.pt', map_location=torch.device('cpu')))
model.eval()

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "✅ Air Quality Prediction Microservice is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = np.array(data['sequence'], dtype=np.float32)

        if sequence.shape != (24, 13):
            return jsonify({"error": "Input must be a 24x13 array"}), 400

        sequence_tensor = torch.tensor(sequence).unsqueeze(0)

        with torch.no_grad():
            prediction = model(sequence_tensor).numpy().flatten().tolist()

        return jsonify({"predicted_values": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
