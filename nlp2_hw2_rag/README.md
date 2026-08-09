# NLP-2 HW2 — RAG (SciFact)

Домашка по Retrieval-Augmented Generation: baseline → chunking → hybrid/rerank → промпт → challenge split.

## Что сделано

- `build_rag_pipeline`: Qdrant + SentenceSplitter / HierarchicalNodeParser, dense / hybrid (BM25 + RRF), опциональный reranker
- конфиги экспериментов: 256 / 1024 / hierarchical, Matryoshka `truncate_dim`, hybrid+rerank, tuned prompt
- заглушки прогонов `run_retrieval` / `run_e2e` на main и challenge (раскомментировать при наличии GPU/моделей)

## Запуск

```bash
cd nlp2_hw2_rag
pip install llama-index-core llama-index-llms-huggingface llama-index-embeddings-huggingface \
  llama-index-vector-stores-qdrant llama-index-retrievers-bm25 \
  qdrant-client ir_datasets ir_measures bitsandbytes accelerate transformers rank-bm25
jupyter notebook nlp2_hw2_rag.ipynb
```

Нужен интернет: SciFact (`ir_datasets`) и веса HF (embedding / LLM / reranker). Без GPU LLM лучше не грузить (`load_llm=False` для retrieval-only).

Артефакты пишутся в `./artifacts/`.
