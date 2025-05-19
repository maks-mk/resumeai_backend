from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import docx  # для чтения .docx файлов

# Загрузка переменных окружения
load_dotenv()

# Инициализация API Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY не найден в переменных окружения")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')  # Используем быструю модель

# Глобальная переменная для хранения текста
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

# Функция для извлечения текста из DOCX
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        # Также извлекаем текст из таблиц, если они есть
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        
        return text
    except Exception as e:
        print(f"Ошибка при чтении DOCX: {e}")
        return ""

# Определяем контекст жизненного цикла приложения
@asynccontextmanager
async def lifespan(app):
    # Код, выполняемый при запуске приложения
    global resume_text
    docx_path = "data2.docx"
    
    if os.path.exists(docx_path):
        resume_text = extract_text_from_docx(docx_path)
        print(f"Текст из резюме (DOCX) загружен, {len(resume_text)} символов")
    else:
        # Запасной вариант - попробовать загрузить PDF, если DOCX не найден
        pdf_path = "data.pdf"
        if os.path.exists(pdf_path):
            resume_text = extract_text_from_pdf(pdf_path)
            print(f"Текст из резюме (PDF) загружен, {len(resume_text)} символов")
        else:
            print(f"Файлы {docx_path} и {pdf_path} не найдены")
    
    yield
    # Код, выполняемый при завершении работы приложения
    # Можно добавить очистку ресурсов, если требуется

# Создаем приложение FastAPI с обработчиком жизненного цикла
app = FastAPI(lifespan=lifespan)

# Настройка CORS для GitHub Pages и локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://maks-mk.github.io",  # GitHub Pages
        "http://localhost:8000",      # Локальная разработка
        "http://127.0.0.1:8000",
        "*"                           # В продакшене лучше убрать и указать конкретные домены
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Класс для запроса
class ChatRequest(BaseModel):
    message: str

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
        
@app.get("/ping")
async def ping():
    return {"status": "ok"}


# Обслуживание статических файлов - должно быть ПОСЛЕ API эндпоинтов
# При деплое на Render этот код нужно закомментировать, так как
# там используется только API без раздачи статики
if os.getenv("ENVIRONMENT") != "production":
    app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000"))) 