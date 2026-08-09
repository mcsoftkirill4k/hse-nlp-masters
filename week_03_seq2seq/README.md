# Week 3 — Seq2seq + attention (семинар)

Семинар по encoder-decoder на переводе RU→EN (описания отелей). BPE, GRU, потом additive attention.

## Что сделано

- токенизация + **BPE** (8000 merge)
- baseline: **BasicModel** (GRU enc/dec без attention)
- `compute_loss` с маской (первый EOS учитываем, паддинг нет)
- **AttentionLayer** (additive) + **AttentiveModel**
- BLEU на dev, визуализация attention (bokeh)

## Запуск

```bash
cd week_03_seq2seq
jupyter notebook week_03_seq2seq_seminar.ipynb
```

Нужны: `torch`, `subword-nmt`, `nltk`, `scikit-learn`, `matplotlib`, `bokeh`.  
Обучение долгое на CPU — на GPU заметно быстрее. `data.txt` и `vocab.py` качаются в ноутбуке.

## Файлы

- `week_03_seq2seq_seminar.ipynb` — семинар с решениями
