# 🕵️ Detecting Information Leakage using Graph based probabilistic reasoning

This is a project that uses a  Graph based probabilistic reasoning model to detect **information leakage** in textual content: Useful in Large language model powered chatbots which uses cache systems for better performance.
Built with **FastAPI**, the API includes endpoints for both prediction and retraining. The model is trained on word-level transitions in documents using a probabilistic approach.

---

## Getting Started

### Clone the Repo
```bash
https://github.com/AkssharRamesh03/Graph-based-Probabilistic-reasoning.git
cd Graph-based-Probabilistic-reasoning
```

### Local Setup (without Docker)
```bash
pip install -r requirements.txt
python training/train_model.py  # Train the model
uvicorn app.main:app --reload  # Start the server at localhost:8000
```

## API Endpoints

### `POST /predict`
Classifies a given piece of text as either leakage or clean.

**Request Body:**
```json
{
  "text": "The home location of Aksshar is SE10 9LS which is 2KM away from the Greenwich NHS Hospital."
}
```
**Response:**
```json
{
  "prediction": "leak"
}
```

### `POST /train`
Retrains the model with custom data.

**Response:**
```json
{
  "status": "Model trained and saved successfully."
}
```

---

## Development Notes
- `model/probabilistic_model.pkl` is where the trained model is saved.
- API automatically reloads model after `/train` is called.

---

## 📁 Project Structure
```
Graph-based-Probabilistic-reasoning/
├── app/
│   ├── main.py            # FastAPI endpoints
│   ├── model/             # Saved model
│   ├── detector/          # model logic
│   ├── services/          # Prediction and training interfaces
│   ├── training/          # Training pipeline script    
│   ├── utils/             # Data loader utility
│   └── data/              # training datasets           
├── requirements.txt       # Dependencies
└── README.md
```
