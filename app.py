from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Инициализация API Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY не найден в переменных окружения")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')  # Используем быструю модель

app = FastAPI()

# Настройка CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Класс для запроса
class ChatRequest(BaseModel):
    message: str

# Глобальная переменная для хранения текста из PDF
resume_text = ""

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Ошибка при чтении PDF: {e}")
        return ""

# Загрузка текста резюме при запуске сервера
@app.on_event("startup")
async def startup_event():
    global resume_text
    pdf_path = "resume.pdf"
    if os.path.exists(pdf_path):
        resume_text = extract_text_from_pdf(pdf_path)
        print(f"Текст из резюме загружен, {len(resume_text)} символов")
    else:
        print(f"Файл {pdf_path} не найден")

# ВАЖНО: API эндпоинты должны быть определены ПЕРЕД монтированием статических файлов!
# Эндпоинт для чата
@app.post("/chat")
async def chat(request: ChatRequest):
    if not resume_text:
        raise HTTPException(status_code=500, detail="Резюме не загружено")
    
    try:
        # Создаем контекст с информацией из резюме
        system_prompt = """Ты AI-ассистент, который помогает посетителям сайта отвечать на вопросы о резюме Колесникова Максима. 
        Отвечай кратко, по делу, дружелюбно. Используй только информацию из резюме.
        
        ВАЖНО: 
        1. Ты отвечаешь о кандидате в третьем лице (он, Максим, кандидат) - а не от его имени.
        2. Посетитель сайта - это потенциальный работодатель, который задает вопросы о кандидате.
        3. Если тебя спрашивают как ты можешь помочь, опиши что ты можешь рассказать о кандидате.
        
        Вот резюме Колесникова Максима:
        """ + resume_text
        
        # Отправляем запрос в Gemini
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [system_prompt]},
                {"role": "user", "parts": [request.message]}
            ]
        )
        
        return {"response": response.text}
    
    except Exception as e:
        print(f"Ошибка API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Обслуживание статических файлов - должно быть ПОСЛЕ API эндпоинтов
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80) 