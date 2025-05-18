from fastapi import FastAPI
from .router import router, lifespan_context

# Создаём экземпляр FastAPI и передаём ему lifespan-контекст
app = FastAPI(lifespan=lifespan_context)
# Подключаем маршруты из router.py
app.include_router(router)