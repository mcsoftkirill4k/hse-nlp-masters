# NLP — магистратура / курсы ВШЭ

Репозиторий с прорешанными семинарами и домашками по NLP (Высшая школа экономики).

Цель: фиксировать практику по курсу (эмбеддинги → RNN/Transformer → RAG/агенты) так, чтобы было видно прогресс обучения.

## Структура

| Папка | Тема |
|-------|------|
| [`week_01_word_embeddings/`](week_01_word_embeddings/) | Word2Vec / FastText / GloVe, OOV, PCA & t-SNE |

## Стек

Python, `nltk`, `gensim`, `scikit-learn`, `bokeh`, позже — PyTorch / HuggingFace / LlamaIndex (по программе курса).

## Как запускать

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Крупные датасеты и веса моделей в git **не** кладутся (см. `.gitignore`).

## Автор

Студент IT / NLP-трека. Материалы курса используются в учебных целях.
