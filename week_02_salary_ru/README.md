# Week 2 mini-research: предсказание зарплаты по вакансиям hh.ru (RU)

Адаптация семинара HSE NLP (salary prediction / TextCNN) под российский рынок IT-вакансий.

| Было в семинаре | Здесь |
|-----------------|--------|
| Train_rev1 (UK, £) | IT-вакансии hh.ru (RUR) |
| `glove-twitter-100` | **Navec** (русские word embeddings, 300d) |
| Title + FullDescription | Title (`name`) + skills / specialization |
| categorical: Company, Location, … | город, опыт, график, тип занятости |

## Зачем

Семинарный пайплайн работает на английских вакансиях и отдаёт фунты. Для портфолио полезнее показать тот же подход на русском: токенизация → словарь → pretrained embeddings → TextCNN → регрессия `log1p(salary)`.

## Данные

Источник: открытый дамп IT-вакансий hh.ru  
[`Kubik91/hac_19_10`](https://github.com/Kubik91/hac_19_10) → `finall_test.csv`  
(~13k строк, все с валютой `RUR`).

Таргет:
- если есть `from` и `to` → среднее;
- иначе доступная граница;
- `gross=True` → грубо `* 0.87` (на руки);
- фильтр выбросов: 20 000 … 400 000 ₽.

> Dream Job / похожие агрегаторы зарплат публичного датасета не отдают; вакансии там часто ведут на hh.ru. Поэтому взят готовый hh-дамп, а не парсинг закрытых страниц отзывов.

## Как запустить

Из корня репозитория:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Navec (~51MB), если ещё нет в artifacts/
curl -L -o week_02_salary_ru/artifacts/navec_hudlit_v1_12B_500K_300d_100q.tar \
  https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

python week_02_salary_ru/train_salary_ru.py
```

Ноутбук с разбором пайплайна: [`salary_prediction_ru.ipynb`](salary_prediction_ru.ipynb).

## Артефакты после обучения

| файл | содержание |
|------|------------|
| `artifacts/metrics.json` | history по epoch, best MAE |
| `artifacts/salary_ru_best.pt` | веса + vocab + DictVectorizer |
| `artifacts/demo_predictions.json` | несколько примеров pred vs true |
| `data/vacancies_clean.csv` | очищенный train-ready датасет |

Крупные файлы (`*.tar`, сырой csv, `.pt`) в git по возможности не коммитятся — см. `.gitignore`.

## Результаты (локальный прогон, CPU, 8 epochs)

| метрика | значение |
|---------|----------|
| clean rows | 12 759 |
| Navec coverage | 78.2% словаря |
| best **MAE ₽** (val) | **≈ 23 600** |
| best mae_log | ≈ 0.33 |
| best mse_log | ≈ 0.18 |

Примеры `demo_predictions.json` (true vs pred) лежат в `artifacts/`.

Это учебный baseline, не прод-модель рынка труда: нет полного HTML-описания вакансии, данные устаревают, вилки на hh часто неполные.

## Структура

```
week_02_salary_ru/
  README.md
  train_salary_ru.py
  salary_prediction_ru.ipynb
  data/
  artifacts/
```
