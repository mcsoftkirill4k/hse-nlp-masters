# Week 5 — Transformers / HuggingFace (семинар)

Семинар: pipelines из `transformers` — sentiment, fill-mask, NER, и как достать tokenizer/model руками.

## Что сделано

- sentiment по девизам домов (GoT) через DistilBERT SST-2
- MLM: год основания СССР (`bert-base-uncased`)
- NER на куске текста про Rosetta / Philae
- tokenizer + `AutoModel`, `pooler_output`
- поправил черновик `MyBertBasedClassifier` (были опечатки в шаблоне)

## Запуск

```bash
cd week_05_transformers_seminar
jupyter notebook week_05_transformers_seminar.ipynb
```

Нужны: `transformers`, `torch`, интернет на первую загрузку моделей.

## Файлы

- `week_05_transformers_seminar.ipynb` — семинар с решениями
