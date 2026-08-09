# Данные

Полный дамп IT-вакансий (~13k строк) лежит локально как `hh_it_vacancies.csv`
(источник: [Kubik91/hac_19_10](https://github.com/Kubik91/hac_19_10/blob/main/finall_test.csv)).

В git коммитится только маленький сэмпл `hh_it_vacancies_sample.csv` для просмотра схемы.

Для обучения скачай полный csv:

```bash
curl -L -o week_02_salary_ru/data/hh_it_vacancies.csv \
  https://raw.githubusercontent.com/Kubik91/hac_19_10/main/finall_test.csv
```

После `python week_02_salary_ru/train_salary_ru.py` появится `vacancies_clean.csv`.
