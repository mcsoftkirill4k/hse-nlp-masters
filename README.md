# NLP — ВШЭ

Семинары и домашки по NLP. Иду по курсу и складываю сюда прорешанное.

## Папки

| Папка | Что внутри |
|-------|------------|
| [`week_01_word_embeddings/`](week_01_word_embeddings/) | Word2Vec / FastText / GloVe, OOV, PCA & t-SNE |
| [`week_02_word_embeddings_hw/`](week_02_word_embeddings_hw/) | FastText, Quora finetune, FAISS |
| [`week_02_salary_ru/`](week_02_salary_ru/) | зарплаты hh.ru + Navec + TextCNN |
| [`week_03_seq2seq/`](week_03_seq2seq/) | encoder-decoder + attention |
| [`week_05_transformers_seminar/`](week_05_transformers_seminar/) | HF pipelines: sentiment / MLM / NER |
| [`week_05_gpt_homework/`](week_05_gpt_homework/) | GPT-2: attention + transformer layer |
| [`week_05_bert_homework/`](week_05_bert_homework/) | BERT / QQP fine-tune |
| [`week_06_peft_homework/`](week_06_peft_homework/) | soft prompts + LoRA |
| [`week_06_rlhf/`](week_06_rlhf/) | reward model + RLHF |
| [`nlp2_hw1_speculative/`](nlp2_hw1_speculative/) | NanoQwen + speculative decoding |
| [`nlp2_hw2_rag/`](nlp2_hw2_rag/) | RAG: chunking, hybrid, rerank |
| [`nlp2_hw3_agents/`](nlp2_hw3_agents/) | LangGraph airline assistant |

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Тяжёлые веса и датасеты в git не кладу (см. `.gitignore`).
