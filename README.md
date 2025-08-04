# 🕵️ Detecting Information Leakage using Graph based probabilistic reasoning

This is a project that uses a  Graph based probabilistic reasoning model to detect **information leakage** in textual content: Usefull in Large language model powered chatbots which uses cache systems for better performance.
Built with **FastAPI**, the API includes endpoints for both prediction and retraining. The model is trained on word-level transitions in documents using a probabilistic approach.

---

## Getting Started

### Clone the Repo
```bash
git clone https://github.com/yourusername/fake_news_api.git
cd fake_news_api
```

### Local Setup (without Docker)
```bash
pip install -r requirements.txt
python training/train_model.py  # Train the model
uvicorn app.main:app --reload  # Start the server at localhost:8000
```

---

## Docker Setup (Recommended)

### 1. Install Docker Desktop
[https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

### 2. Build the Image
```bash
docker build -t fake-news-api .
```

### 3. Run the Container
```bash
docker run -p 8000:8000 fake-news-api
```

Server will be live at: [http://localhost:8000](http://localhost:8000)

---

## API Endpoints

### `POST /predict`
Classifies a given piece of text as either leakage or clean.

**Request Body:**
```json
{
  "text": "North Korea Claims to Have Invented Invisible Bomb"
}
```
**Response:**
```json
{
  "prediction": "fake"
}
```

### `POST /train`
Retrains the model with new data.

**Request Body:**
```json
{
  "real_news": ["UK economy grows", "President addresses the nation"],
  "fake_news": ["Aliens spotted in parliament", "Cure for death found"]
}
```
**Response:**
```json
{
  "status": "Model trained and saved successfully."
}
```

---

## Development Notes
- `model/markov_model.pkl` is where the trained model is saved.
- API automatically reloads model after `/train` is called.
- Training is word-token based; punctuation and casing matter unless further cleaned.

---

## 📁 Project Structure
```
fake_news_api/
├── app/
│   ├── main.py            # FastAPI endpoints
│   ├── models/            # Markov model logic
│   ├── services/          # Prediction and training interfaces
│   ├── utils/             # Data loader utility
│   └── data/              # Example training datasets
├── training/              # Training pipeline script
├── model/                 # Saved model
├── Dockerfile             # For containerization
├── requirements.txt       # Dependencies
└── README.md
```
