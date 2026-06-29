# Customer Churn Prediction

Предсказание оттока клиентов телеком-компании с акцентом на **бизнес-метрики**, а не только на accuracy.

## Задача

Построить модель, которая заранее выявляет клиентов с высокой вероятностью оттока (Churn), чтобы команда retention могла точечно воздействовать на них до потери LTV.

## Датасет

- **Источник:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Объём:** ~7 000 клиентов, 21 признак (демография, тариф, услуги, платежи)
- **Целевая переменная:** `Churn` (Yes / No), дисбаланс классов ~27% оттока

## Подход

1. **EDA** — очистка `TotalCharges`, удаление записей с `tenure = 0`, анализ категориальных признаков
2. **Feature engineering** — One-Hot Encoding категорий, числовые признаки без масштабирования (для tree-based моделей)
3. **Сравнение моделей** — Dummy → Logistic Regression → Random Forest → XGBoost (5-fold Stratified CV)
4. **Тюнинг** — RandomizedSearchCV по ROC-AUC и отдельная модель с оптимизацией по **Recall**
5. **Бизнес-оценка** — расчёт ожидаемой прибыли при разных порогах и допущениях (LTV, стоимость контакта, вероятность удержания)
6. **Интерпретация** — SHAP для понимания драйверов оттока

## Результаты

| Модель | ROC-AUC (CV) | Accuracy (test) | Recall (test) |
|--------|-------------|-----------------|---------------|
| Baseline | 0.50 | — | — |
| Logistic Regression | **0.847** | — | — |
| XGBoost (ROC-AUC) | 0.850 | 0.78 | 0.51 |
| XGBoost (Recall) | 0.847 | 0.79 | **0.76** |

**Ключевой инсайт:** модель с лучшим ROC-AUC давала **отрицательную** ожидаемую прибыль (−34.89 $/клиент), тогда как recall-оптимизированная модель — **+2.12 $/клиент** при заданных бизнес-допущениях.

## Стек

`pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn`

## Запуск

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook "customer churn prediction.ipynb"
```

## Файлы

| Файл | Описание |
|------|----------|
| `customer churn prediction.ipynb` | Основной ноутбук с EDA, моделями и бизнес-анализом |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Исходные данные |
| `requirements.txt` | Зависимости Python |
