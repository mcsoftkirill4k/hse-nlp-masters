# NLP-2 HW1 — Speculative decoding / NanoQwen

Собираем куски Qwen (RMSNorm, RoPE, SwiGLU, attention block) и жадный speculative decoding: draft предлагает K токенов, target принимает префикс совпадений.

## Что сделано

- `QwenRMSNorm`, `apply_rope`, `QwenMLP` (SwiGLU), `NanoQwenBlock` + `QwenAttention`
- `speculative_sampling` (accept/reject) + юнит-тест на mock-последовательности
- бенчмарк с `HeavyTarget` (искусственная задержка)

## Запуск

```bash
cd nlp2_hw1_speculative
pip install transformers accelerate safetensors sentencepiece
jupyter notebook nlp2_hw1_speculative.ipynb
```

Первый запуск тянет `Qwen2.5-1.5B` и `0.5B` с HuggingFace — нужен диск и лучше GPU.
