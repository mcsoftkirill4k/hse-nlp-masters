# Week 1 — Word Embeddings (семинар)

Прорешанный семинар по word embeddings + мои комментарии и тесты.

## Что сделано

- токенизация Quora через `WordPunctTokenizer`
- обучение **FastText** и **Word2Vec** на корпусе
- **тест OOV:** Word2Vec → `KeyError` на неизвестном слове; FastText → вектор через char n-grams
- pretrained **GloVe Twitter 100d** (`gensim.downloader`)
- визуализация: **PCA** и **t-SNE** (bokeh)
- заметка: **`StandardScaler` до PCA**, не после (методология)
- краткое сравнение архитектур Word2Vec / GloVe / FastText и чтение графиков PCA vs t-SNE

## Запуск

```bash
cd week_01_word_embeddings
jupyter notebook week_01_word_embeddings_seminar.ipynb
```

Первый запуск скачает `quora.txt` и модель GloVe (нужен интернет).

## Файлы

- `week_01_word_embeddings_seminar.ipynb` — основной ноутбук
