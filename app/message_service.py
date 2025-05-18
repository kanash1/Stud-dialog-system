import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, asc
from app.models import Message

# Настройка логирования
logger = logging.getLogger(__name__)

# Получение последних N сообщений по ID сессии
async def get_last_n_messages_by_session_id(
    session: AsyncSession,
    session_id: int,
    n: int
):
    logger.info(f"Fetching last {n} messages for session_id={session_id}")
    result = await session.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(asc(Message.created_at), asc(Message.message_id))
        .limit(n)
    )
    messages = result.scalars().all()
    logger.info(f"Fetched {len(messages)} messages")
    return messages

# Создание нового сообщения и его сохранение в БД
async def create_message(
    session: AsyncSession,
    session_id: int,
    message: dict[str, str]
) -> Message:
    logger.info(f"Creating message for session_id={session_id}")
    new_message = Message(session_id=session_id, message=message)
    session.add(new_message)
    await session.flush()  # Синхронизация с БД (но без commit)
    logger.info(f"Message created with id={new_message.message_id}")
    return new_message

# Получение сообщений с постраничной навигацией (cursor-based)
async def get_messages_by_session_id_cursor(
    session: AsyncSession,
    session_id: int,
    limit: int = 20,
    after_id: int | None = None
) -> list[Message]:
    logger.info(
        f"Fetching messages for sid={session_id} after_id={after_id} limit={limit}"
    )

    query = select(Message).where(Message.session_id == session_id)

    if after_id is not None:
        # Добавление условия для cursor-based пагинации
        query = query.where(Message.message_id > after_id)

    query = query.order_by(asc(Message.message_id)).limit(limit)

    result = await session.execute(query)
    messages = result.scalars().all()

    logger.info(f"Fetched {len(messages)} messages")
    return messages