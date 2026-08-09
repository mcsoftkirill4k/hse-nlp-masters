# Week 6 — RLHF

Что сделал: reward model на IMDB (positive), pairwise accuracy, reward-guided generation, плюс скелет main assignment (вариант C — reward по длине).

Запуск:
```bash
cd week_06_rlhf
jupyter notebook week_06_rlhf.ipynb
```

Нужны GPU и пакеты из первой ячейки (`trl` / `peft` / `transformers`). PPO и reward training тяжёлые — код рабочий, но считать долго.
