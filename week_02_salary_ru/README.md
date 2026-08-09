# Week 2 — зарплата по вакансиям hh.ru

Тот же пайплайн, что в семинаре по salary prediction, но на русских IT-вакансиях.

## Что поменял относительно семинара

- данные: hh.ru (RUR), не UK Kaggle (£)
- эмбеддинги: Navec 300d вместо GloVe Twitter
- текст: название вакансии + skills
- категории: город, опыт, график, тип занятости
- таргет: `log1p` от зарплаты в ₽

## Данные

Взял открытый дамп IT-вакансий: [Kubik91/hac_19_10](https://github.com/Kubik91/hac_19_10) (`finall_test.csv`).

Зарплату считаю так: среднее from/to, если gross — примерно `* 0.87`, выбросы режу 20k–400k ₽.

У Dream Job нормального публичного датасета нет, поэтому hh-дамп.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

curl -L -o week_02_salary_ru/artifacts/navec_hudlit_v1_12B_500K_300d_100q.tar \
  https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

python week_02_salary_ru/train_salary_ru.py
```

Ноутбук: `salary_prediction_ru.ipynb`.

## Результат (CPU, 8 epoch)

- ~12.7k вакансий после чистки
- покрытие словаря Navec ~78%
- MAE на val ≈ **23.6k ₽**

Это учебный baseline, не оценка рынка «как есть».

## Файлы

- `train_salary_ru.py` — обучение
- `salary_prediction_ru.ipynb` — разбор
- `data/` — данные / sample
- `artifacts/` — метрики и веса (крупное в git не кладу)
