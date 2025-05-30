import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, asc
from .models import Session

# Настройка логирования
logger = logging.getLogger(__name__)

# Получение сессии по session_id и user_id
async def get_session_by_sid_and_uid(
    session: AsyncSession,
    session_id: int,
    user_id: int
) -> Session | None:
    logger.info(f"Fetching session with sid={session_id}, uid={user_id}")
    
    # Выполняем запрос на получение сессии по идентификаторам
    result = await session.execute(
        select(Session).where(
            Session.session_id == session_id,
            Session.user_id == user_id
        )
    )
    
    session_obj = result.scalar_one_or_none()
    if session_obj:
        logger.info(f"Session found: {session_obj.session_id}")
    else:
        logger.info("Session not found.")
    
    return session_obj

# Создание новой сессии для пользователя
async def create_session(
    session: AsyncSession,
    user_id: int
) -> Session:
    logger.info(f"Creating new session for user_id={user_id}")
    
    user_session = Session(user_id=user_id)
    session.add(user_session)
    
    # Сохраняем в БД, но не коммитим (коммит снаружи)
    await session.flush()
    
    logger.info(f"Session created with session_id={user_session.session_id}")
    return user_session

# Получение всех сессий пользователя с постраничной навигацией (cursor-based)
async def get_sessions_by_user_id_cursor(
    session: AsyncSession,
    user_id: int,
    limit: int = 20,
    after_id: int | None = None
) -> list[Session]:
    logger.info(
        f"Fetching sessions for user_id={user_id} after_id={after_id} limit={limit}" 
    )

    query = select(Session).where(Session.user_id == user_id)

    if after_id is not None:
        # Добавление условия для cursor-based пагинации
        query = query.where(Session.session_id > after_id)

    query = query.order_by(asc(Session.session_id)).limit(limit)

    result = await session.execute(query)
    sessions = result.scalars().all()

    logger.info(f"Fetched {len(sessions)} sessions.")
    return sessions