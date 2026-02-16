## Прогнозирование оттока клиентов (Churn Prediction)

Это мой учебный проект по машинному обучению. Здесь я попробовал построить модель, которая предсказывает, уйдёт ли клиент телеком-компании (отток) или останется. Для удобства я сделал небольшой веб-сервис на FastAPI: можно загрузить данные клиента в формате JSON и сразу получить ответ — склонен он к оттоку или нет.

## Что делает проект

- Обучает модель на данных `Telco_Churn.csv`.
- Использует XGBoost + SMOTE для борьбы с дисбалансом классов.
- Сохраняет лучшую модель в файл `best_model.pkl`.
- Запускает веб-сервер с формой для загрузки JSON.
- По загруженным данным выдаёт предсказание и вероятность.

## Какие технологии использовал

- **Python** (3.9+)
- **FastAPI** 
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn**
- **Joblib**
- **Docker**


## Как запустить проект локально

### 1. Склонировать репозиторий

```bash
git clone https://github.com/alexkaaz/Churn_prediction.git
cd Churn_prediction.git
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv
source venv/bin/activate  # на Linux/Mac
# или venv\Scripts\activate на Windows

pip install -r requirements.txt
```

### 2.1 Обучить модель заново 

Если есть желание поменять саму модель или ее гиперпараметры:

```bash
python train.py
```

Скрипт прочитает данные, подберёт гиперпараметры с помощью GridSearchCV и сохранит лучшую модель в `best_model.pkl`.

### 4. Запустить веб-сервер

```bash
uvicorn app:app --reload
```

После запуска открой браузер и перейди по адресу: http://127.0.0.1:8000

## Как пользоваться

1. На главной странице будет форма с кнопкой «Выберите файл» (или можно перетащить файл мышкой).
2. Нужно выбрать JSON-файл с данными клиента. Шаблон такого файла:

```json
{
    "gender": "Male"/"Famale",
    "SeniorCitizen": 0/1,
    "Partner": "Yes"/"No",
    "Dependents": "yes"/"No",
    "tenure": int64,
    "PhoneService": "Yes"/"No"/"No phone Service",
    "MultipleLines": "Yes"/"No"/"No phone Service",
    "InternetService":  Fiber optic/DSL/No,
    "OnlineSecurity": "Yes"/"No"/"No phone Service",
    "OnlineBackup": "Yes"/"No"/"No phone Service",
    "DeviceProtection": "Yes"/"No"/"No phone Service",
    "TechSupport": "Yes"/"No"/"No phone Service",
    "StreamingTV": "Yes"/"No"/"No phone Service",
    "StreamingMovies": "Yes"/"No"/"No phone Service",
    "Contract": "Month-to-month"/"Two year"/"One Year",
    "PaperlessBilling": "Yes"/"No",
    "PaymentMethod": "Electronic check"/"Mailed check"/"Bank transfer (automatic)"/"Credit card (automatic)",
    "MonthlyCharges": float64,
    "TotalCharges": float64
}
```

3. После загрузки нажми «Загрузить файл».
4. Сервер обработает данные, сохранит копию запроса в папку `logs/` и покажет результат:
   - **Склонен к оттоку** или **Не склонен к оттоку**
   - **Вероятность** в процентах
   - Имя сохранённого файла и время запроса.

5. Можно нажать «Вернуться к форме» и попробовать другой файл.

## Как запустить через Docker

Если хочешь запустить в контейнере:

```bash
docker build -t churn-app .
docker run -p 8000:8000 churn-app
```

После этого приложение будет доступно по тому же адресу http://localhost:8000.

## Нюансы

- Проект учебный, поэтому не рассчитан на большие нагрузки.
- JSON должен содержать **все признаки**, кроме `Churn`. Порядок полей не важен.
- Если какое-то поле отсутствует или имеет неправильный тип, модель может выдать ошибку (пока что обработка минимальная).
- Модель обучена на данных Telco Customer Churn, поэтому признаки должны точно соответствовать исходным.

Если есть вопросы или предложения — пишите! :)
