# Week 2 — FastText + поиск по Quora (домашка)

Прорешанная домашка: эмбеддинги на «Войне и мире», дообучение на Quora, поиск похожих вопросов.

## Что сделано

- чистка Gutenberg-текста + токенизация в предложения
- обучение **FastText** (skip-gram, dim=100, α=3e-2)
- finetune на Quora с низким lr, чтобы не забыть старый контекст
- эмбеддинг предложения = mean pooling + L2-норма
- поиск: **numpy** (точный) и **FAISS HNSW** (быстрее)

## Запуск

```bash
cd week_02_word_embeddings_hw
jupyter notebook week_02_word_embeddings_hw.ipynb
```

Нужны: `gensim`, `nltk`, `unidecode`, `faiss-cpu`, `numpy`. Первый запуск качает War and Peace и Quora.

## Файлы

- `week_02_word_embeddings_hw.ipynb` — ноутбук с решениями
