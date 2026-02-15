# Sentiment-Transformer-IMDB

# Transformer-Based Sentiment Classification (IMDB)

## Overview
This project implements a Transformer-based sentiment classification model from scratch using PyTorch.  
The model is trained on the IMDB movie review dataset to classify reviews as **positive** or **negative**.

The project covers the complete pipeline including:
- Data exploration and preprocessing
- Transformer model implementation
- Training and evaluation
- Inference on custom reviews
- Model deployment using FastAPI
- Basic front-end interface for prediction

---

## Model Architecture
The model consists of:
- Token embedding layer
- Positional encoding
- Multi-head self-attention encoder blocks
- Feed-forward network
- Masked mean pooling
- Linear classification layer

---

## Results
The model achieves approximately **85% validation accuracy**, showing good performance on sentiment classification.

---

## Deployment
The trained model is deployed using **FastAPI**.  
The API accepts a movie review and returns the predicted sentiment.

### Run the API
```bash
pip install -r requirements.txt
uvicorn api:app --reload
