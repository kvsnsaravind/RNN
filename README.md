# RNN

# 📖 RNN-Based Text Generator on Harry Potter Book 1

This project implements a **Recurrent Neural Network (RNN)** in TensorFlow/Keras to generate text in the style of *Harry Potter and the Philosopher's Stone*. The model learns from the first book's text and predicts the next word based on a given input sequence.

---

## 📂 Dataset

- **Source**: [Harry Potter Book 1](https://www.kaggle.com/datasets) (you must download and place `hp_1.txt` in your working directory)
- **Preprocessing**:
  - Lowercased the text
  - Tokenized words using Keras's `Tokenizer`
  - Converted words to integer sequences
  - Created input sequences of 50 words each to predict the 51st word

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts word indices into dense vectors
- **SimpleRNN Layer**: Processes sequences and learns temporal dependencies
- **Dense Layer**: Adds non-linearity
- **Output Layer**: Softmax over the entire vocabulary to predict the next word

```python
model = Sequential([
    Embedding(input_dim=total_words, output_dim=64, input_length=50),
    SimpleRNN(256),
    Dense(256, activation='relu'),
    Dense(total_words, activation='softmax')
])
```
- Loss Function: categorical_crossentropy

- Optimizer: adam

- Training Epochs: 10 (Change all these and try)

## 📊 Notes
- This model uses one-hot encoding for labels.

- You can try replacing SimpleRNN with LSTM or GRU for better performance on longer contexts.

- The model is relatively small and good for experimentation or educational purposes.

- The vocabulary size depends on the unique words in the dataset (usually ~20K+).

## 📌 Future Improvements
- Add temperature sampling for more diverse text generation

- Replace RNN with LSTM or Transformer-based models

- Fine-tune on multiple books or a larger corpus

- Use pre-trained embeddings like GloVe or Word2Vec

## 📜 License
This project is for educational use only. The Harry Potter dataset should not be used for commercial purposes.
