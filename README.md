# Fake News Classifier API

This is a production-ready backend API that uses a custom-built Markov Chain model to classify news articles as **real** or **fake**.

Built with **FastAPI**, the API includes endpoints for both prediction and model retraining. The model is trained on word-level transitions in news text using a probabilistic Markov Chain approach.

---

## 📦 Features

- 🧠 Markov Chain–based text classification
- 🔁 API-based retraining endpoint
- 🚀 FastAPI backend
- 🐳 Dockerized for deployment
- 🔬 Supports JSON input/output for easy integration

---

## 🚀 Getting Started

### 📥 Clone the Repo
```bash
git clone https://github.com/yourusername/fake_news_api.git
cd fake_news_api
```

### 🐍 Local Setup (without Docker)
```bash
pip install -r requirements.txt
python training/train_model.py  # Train the model
uvicorn app.main:app --reload  # Start the server at localhost:8000
```

---

## 🐳 Docker Setup (Recommended)

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

## 🔌 API Endpoints

### 📍 `POST /predict`
Classifies a single news text.

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

### 📍 `POST /train`
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

## 🧪 Development Notes
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

---

## 🛡️ License
This project is licensed under the MIT License.

---

## ✨ Future Improvements
- Add authentication to protect training endpoint
- Store uploaded training data for audit
- Enable logging and model evaluation metrics
- Swap Markov with transformer-based model as an upgrade

---

Made with 💻 + ⚙️ by [Your Name]
