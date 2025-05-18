import logging
from fastapi import (
    APIRouter, HTTPException, Request, Depends,
    File, UploadFile, FastAPI, Query
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from jose import JWTError, jwt, ExpiredSignatureError
from datetime import timedelta, datetime, timezone
from typing import AsyncGenerator, List, Optional
from types import SimpleNamespace
from contextlib import asynccontextmanager

from .rag_chain import (
    RAGSystem, ReadFileError,
    UnknownAdapterError, UnsupportedFileFormatError
)
from .models import User, Base
from .shemas import (
    MessageResponse, PaginatedResponse, QueryRequest, RegisterRequest,
    LoginRequest, SessionResponse, TokenResponse, LLMResponse
)
from . import session_service, message_service, user_service, config

# --- Логирование ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Инициализация маршрутов ---
router = APIRouter()

# --- Объекты движка и сессии ---
engine = create_async_engine(config.DATABASE_URL, echo=True)
async_session_maker = async_sessionmaker(
    bind=engine, expire_on_commit=False
)

# --- Lifespan (инициализация/деинициализация компонентов) ---
@asynccontextmanager
async def lifespan_context(app: FastAPI):
    logger.info("App starting... initializing DB and RAG system")
    await create_db_and_tables()
    rag_system = RAGSystem(
        model=config.MODEL,
        adapters=list(config.LORA_MODULES.keys())
    )
    await rag_system.init_system(engine)
    app.state.store = SimpleNamespace()
    app.state.store.rag_system = rag_system
    logger.info("RAG system initialized")
    yield
    logger.info("App shutting down")

# --- Генератор сессий для Depends ---
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

# --- Получение RAGSystem из состояния приложения ---
def get_rag_system(request: Request) -> RAGSystem:
    rag_system = request.app.state.store.rag_system
    if rag_system is None:
        raise RuntimeError("RAGSystem not initialized")
    return rag_system

# --- Создание таблиц при старте ---
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("Database and extensions initialized.")

# --- Создание JWT токена ---
def create_access_token(
    data: dict, expires_delta: timedelta | None = None
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)

# --- Регистрация пользователя ---
@router.post("/register")
async def register_user(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session)
) -> dict:
    existing_user = await user_service.get_user_by_username(
        session, request.name
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    await user_service.create_user(session, request.name, request.password)
    await session.commit()
    logger.info(f"User registered: {request.name}")
    return {"message": "User registered successfully"}

# --- Аутентификация пользователя ---
@router.post("/login")
async def login_user(
    request: LoginRequest,
    session: AsyncSession = Depends(get_async_session)
) -> TokenResponse:
    user = await user_service.get_user_by_username(session, request.name)
    if not user or not user_service.verify_password(user, request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": str(user.user_id)})
    logger.info(f"User logged in: {request.name}")
    return TokenResponse(token=access_token)

# --- Получение текущего пользователя из JWT ---
async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_async_session)
) -> User:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    token = auth.split("Bearer ")[1]
    try:
        payload = jwt.decode(
            token, config.SECRET_KEY, algorithms=[config.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,detail="Invalid token payload"
            )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await user_service.get_user_by_id(session, int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Запрос к модели ---
@router.post("/query")
async def query_model(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    rag_system: RAGSystem = Depends(get_rag_system)
) -> LLMResponse:
    try:
        if request.session_id:
            user_session = await session_service.get_session_by_sid_and_uid(
                session, request.session_id, user.user_id
            )
            if user_session is None:
                raise HTTPException(
                    status_code=404, detail="Session not found"
                )
        else:
            user_session = await session_service.create_session(
                session, user.user_id
            )

        messages = await message_service.get_last_n_messages_by_session_id(
            session, user_session.session_id, 10
        )
        chat_history = [m.message for m in messages]

        user_message = await message_service.create_message(
            session, user_session.session_id,
            {"type": "human", "content": request.query}
        )

        response_text = await rag_system.invoke(
            input={
                "input": request.query,
                "chat_history": chat_history,
            },
            use_rag=request.use_rag,
            adapter=request.adapter
        )

        agent_message = await message_service.create_message(
            session, user_session.session_id,
            {"type": "ai", "content": response_text}
        )
        await session.commit()

        logger.info(f"User {user.user_id} made query: {request.query}")

        return LLMResponse(
            response=response_text,
            session_id=user_session.session_id,
            user_message_id=user_message.message_id,
            agent_message_id=agent_message.message_id
        )
    except UnknownAdapterError as e :
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=404, detail="Unkown adapter Error"
        )
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        )

# --- Загрузка документов ---
@router.post("/upload")
async def upload_docs(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    rag_system: RAGSystem = Depends(get_rag_system)
):
    file_tuples = [
        (file.filename, await file.read()) for file in files
    ]
    conn = await session.connection()
    try:
        await rag_system.document_store.add_documents(
            file_tuples, engine=conn
        )
        await session.commit()
        logger.info(
            f"Indexed {len(files)} file(s): {[f.filename for f in files]}"
        )
        return {
            "status": "indexed",
            "files": [file.filename for file in files]
        }
    except ReadFileError as e:
        raise HTTPException(
            status_code=400, detail=f"{e}"
        )
    except UnsupportedFileFormatError as e:
        raise HTTPException(
            status_code=415, detail=str(e)
        )

# --- Получение списка сессий пользователя ---    
@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    sessions = await session_service.get_sessions_by_user_id(session, user.user_id)
    return sessions


# --- Получение списка сообщений из сесиии пользователя ---   
@router.get("/messages/{session_id}", response_model=PaginatedResponse[MessageResponse])
async def get_session_messages(
    session_id: int,
    limit: int = Query(20, ge=1, le=100),
    after_id: Optional[int] = Query(None),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    user_session = await session_service.get_session_by_sid_and_uid(
        session, session_id, user.user_id
    )
    if user_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await message_service.get_messages_by_session_id_cursor(
        session=session,
        session_id=session_id,
        limit=limit,
        after_id=after_id
    )

    next_cursor = messages[-1].message_id if messages else None

    return PaginatedResponse[MessageResponse](
        items=[MessageResponse.from_orm(m) for m in messages],
        next_cursor=next_cursor
    )