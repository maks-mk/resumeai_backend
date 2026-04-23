import os
import logging
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
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

# 2. Проверка API ключа NVIDIA
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY обязателен. Проверьте файл .env")

# Обновляем название модели по умолчанию
MODEL_NAME = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
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
    Ты — ИИ-ассистент на портфолио Максима Колесникова. Твоя цель — продать его навыки так, чтобы рекрутеру было интересно читать. 
    Твой стиль: дружелюбный, уверенный и лаконичный (как коллега-айтишник на кофе-брейке). Никакого канцелярита и воды!

    Строгие правила:
    1. СРАЗУ к делу. Без «Здравствуйте, отличный вопрос!» или «Я искусственный интеллект...».
    2. Максим — это «он». Ты — его цифровой бро/помощник.
    3. Опирайся ТОЛЬКО на данные ниже. Если информации нет, не извиняйся. Просто креативно отшутись, что твоя база об этом умалчивает, и предложи написать Максиму лично. Каждый раз придумывай новую формулировку для отказа.
    4. Разделяй текст на короткие абзацы. Используй списки. Никаких эмодзи, никаких таблиц, никаких длинных простыней текста.

    Секретные материалы (резюме):
    {full_context_text}
    """    
    try:
        # Инициализируем клиента OpenAI, указывая API NVIDIA как base_url
        client = AsyncOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        
        # Сохраняем клиент и инструкцию в состояние приложения
        app.state.client = client
        app.state.system_instruction = system_instruction
        
        logger.info(f"NVIDIA API Клиент ({MODEL_NAME}) инициализирован. Контекст загружен.")

    except Exception as e:
        logger.error(f"Ошибка инициализации NVIDIA API: {e}")

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

    client: AsyncOpenAI = req.app.state.client
    prompt = request.message

    # Асинхронный генератор, который отдает текст по кусочкам
    async def stream_generator():
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": req.app.state.system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
                presence_penalty=0.4,     # Штрафует за заезженные фразы, заставляя использовать новые слова
                frequency_penalty=0.3,    # Снижает вероятность повторения одних и тех же конструкций
                stream=True,
                max_tokens=1024 # Можно изменить при необходимости
            )
            
            async for chunk in response:
                # В структуре ответа OpenAI/NVIDIA текст находится в delta.content
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Ошибка при потоковой генерации: {e}")
            yield "\n[Системная ошибка: Не удалось завершить поток данных]"

    # Возвращаем потоковый ответ (используем асинхронный генератор)
    return StreamingResponse(stream_generator(), media_type="text/plain")
    

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok", "model": MODEL_NAME, "provider": "nvidia"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)