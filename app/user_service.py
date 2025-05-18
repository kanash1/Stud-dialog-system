import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.hash import bcrypt

from app.models import User

# --- Логирование ---
logger = logging.getLogger(__name__)

# --- Получить пользователя по имени пользователя ---
async def get_user_by_username(
    session: AsyncSession,
    username: str
) -> User | None:
    logger.info(f"Searching for user by username: {username}")
    result = await session.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()
    if user:
        logger.info("User found by username")
    else:
        logger.info("User not found by username")
    return user

# --- Получить пользователя по ID ---
async def get_user_by_id(
    session: AsyncSession,
    id: int
) -> User | None:
    logger.info(f"Searching for user by ID: {id}")
    result = await session.execute(
        select(User).where(User.user_id == id)
    )
    user = result.scalar_one_or_none()
    if user:
        logger.info("User found by ID")
    else:
        logger.warning("User not found by ID")
    return user

# --- Хеширование пароля ---
def hash_password(password: str) -> str:
    logger.debug("Hashing password")
    return bcrypt.hash(password)

# --- Проверка пароля ---
def verify_password(user: User, password: str) -> bool:
    logger.debug("Verifying password")
    return bcrypt.verify(password, user.password_hash)

# --- Создание нового пользователя ---
async def create_user(
    session: AsyncSession,
    username: str,
    password: str
) -> User:
    logger.info(f"Creating new user: {username}")
    password_hash = hash_password(password)
    user = User(
        username=username,
        password_hash=password_hash
    )
    session.add(user)
    await session.flush()
    logger.info(f"User created: {username}")
    return user