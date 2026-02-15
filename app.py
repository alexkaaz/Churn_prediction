from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import json
import joblib
import os
from datetime import datetime

app = FastAPI()

model_path = "best_model.pkl"
model = None

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        print("Модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
else:
    print(f"Файл {model_path} не найден")

HTML_FORM = 'templates/index.html'

@app.get("/", response_class=HTMLResponse)
async def form_page():
    return FileResponse(HTML_FORM)

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)
    X = pd.DataFrame([data.copy()])
    
    # Сохраняем данные запроса 
    timestamp = datetime.now().isoformat()
    data['submitted_at'] = timestamp
    filename = f"logs\submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Получаем предсказание модели
    prediction = None
    probability = None
    
    try:
        proba = model.predict_proba(X)[0]
        prediction = int(model.predict(X)[0])
        probability = float(max(proba))
    except Exception as e:
        prediction = "error"
        probability = "error"
    
    # Генерируем HTML-страницу с результатом
    # Определяем текст предсказания
    if prediction == 1:
        pred_label = "Склонен к оттоку"
        pred_color = "#cf0700"
    elif prediction == 0:
        pred_label = "Не склонен к оттоку"
        pred_color = "#05a705"
    else:
        pred_label = "Не удалось получить предсказание"
        pred_color = "#000000"
    
    html_result = f"""<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>Результат предсказания</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
                .card {{ background: #f9f9f9; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; margin-top: 0; }}
                .prediction {{ font-size: 28px; font-weight: bold; color: {pred_color}; margin: 20px 0; }}
                .probability {{ font-size: 18px; color: #555; }}
                .details {{ background: #eef; padding: 15px; border-radius: 6px; margin: 25px 0; }}
                .button {{ display: inline-block; padding: 10px 20px; background: #0077cc; color: white; text-decoration: none; border-radius: 4px; }}
                .button:hover {{ background: #005fa3; }}
                .error {{ color: #d9534f; background: #f2dede; padding: 10px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Результат анализа клиента</h1>
                <div class="prediction">{pred_label}</div>
                <div class="probability">Вероятность: <strong>{probability:.1%}</strong></div>'
    
                <div class="details">
                    <p><strong>Данные сохранены в файл:</strong> {filename}</p>
                    <p><strong>Время отправки:</strong> {timestamp}</p>
                </div>
                <a href="/" class="button">Вернуться к форме</a>
            </div>
        </body>
        </html>"""
    
    return HTMLResponse(content=html_result)
