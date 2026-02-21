import os
import logging
from contextlib import asynccontextmanager

from google import genai
import uvicorn
import anyio
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# 2. Проверка API ключа
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY обязателен. Проверьте файл .env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
DATA_DIR = "data"


def load_file(name: str) -> str:
    """Загружает текст из файла в папке data."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        logger.warning(f"Файл не найден: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Ошибка чтения {path}: {e}")
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружает контекст и инициализирует AI при старте приложения.
    Здесь мы собираем ВСЁ резюме в одну строку для максимального качества ответов.
    """
    # Собираем полный текст резюме из всех файлов
    files_to_load = [
        "core.txt", 
        "contacts.txt", 
        "experience.txt", 
        "skills.txt", 
        "projects.txt",
        "education.txt",  
        "about.txt"       
    ]
    full_resume_content = []

    for filename in files_to_load:
        content = load_file(filename)
        if content:
            full_resume_content.append(content)
    
    full_context_text = "\n\n".join(full_resume_content)

    if not full_context_text:
        logger.error("Критическая ошибка: Не удалось загрузить файлы резюме.")
        full_context_text = "Информация о кандидате временно недоступна."

    # Настраиваем личность бота
    system_instruction = f"""
    Ты — профессиональный и дружелюбный AI-ассистент на сайте-портфолио Максима Колесникова.
    
    ТВОЯ ЗАДАЧА:
    Отвечать на вопросы посетителей (рекрутеров, коллег), используя ИСКЛЮЧИТЕЛЬНО предоставленный текст резюме.

    ПРАВИЛА ПОВЕДЕНИЯ:
    1. **Без лишних приветствий:** Если пользователь задал конкретный вопрос (например: "Какой стек?", "Где работал?"), ОТВЕЧАЙ СРАЗУ ПО СУТИ. Не пиши "Здравствуйте" и не представляйся, если вопрос подразумевает получение факта.
    2. **Приветствие:** Здоровайся, ТОЛЬКО если сообщение пользователя состоит только из приветствия ("Привет", "Добрый день").
    3. **Тон:** Общайся вежливо, уверенно, но без лишнего официоза. Используй третье лицо ("Максим умеет", "Он работал").
    4. **Честность:** Не выдумывай факты. Если информации нет в тексте, скажи: "В резюме это не указано, но вы можете спросить у Максима лично".
    5. **Формат:** Старайся отвечать связным текстом, используй списки только когда перечисляешь много пунктов.

    ПОЛНОЕ РЕЗЮМЕ МАКСИМА:
    {full_context_text}
    """

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Сохраняем клиент и инструкцию в состояние приложения
        app.state.client = client
        app.state.system_instruction = system_instruction
        
        logger.info(f"Клиент Gemini ({MODEL_NAME}) инициализирован. Контекст загружен.")

    except Exception as e:
        logger.error(f"Ошибка инициализации Gemini: {e}")

    yield

    logger.info("Приложение останавливается.")


app = FastAPI(title="Resume Chatbot API", lifespan=lifespan)

# Разрешаем запросы с вашего сайта и локальной машины
origins = [
    "https://maks-mk.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    if not hasattr(req.app.state, "client"):
        raise HTTPException(status_code=503, detail="AI сервис не инициализирован")

    client = req.app.state.client
    prompt = request.message

    # Создаем генератор, который будет по кусочкам отдавать текст из Gemini
    def stream_generator():
        try:
            # Используем метод потоковой генерации (stream)
            response = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "system_instruction": req.app.state.system_instruction,
                    "temperature": 0.6 
                }
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Ошибка при потоковой генерации: {e}")
            yield "\n[Системная ошибка: Не удалось завершить поток данных]"

    # Возвращаем потоковый ответ
    return StreamingResponse(stream_generator(), media_type="text/plain")
    

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)