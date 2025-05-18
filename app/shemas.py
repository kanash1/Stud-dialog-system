from pydantic import BaseModel
from pydantic.generics import GenericModel
from typing import Generic, TypeVar, List, Optional

class RegisterRequest(BaseModel):
    name: str
    password: str

class LoginRequest(BaseModel):
    name: str
    password: str

class TokenResponse(BaseModel):
    token: str

class QueryRequest(BaseModel):
    query: str
    use_rag: bool
    session_id: Optional[int] = None
    adapter: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    session_id: int
    user_message_id: int
    agent_message_id: int

class MessageResponse(BaseModel):
    message_id: int
    session_id: int
    message: dict[str, str]

    class Config:
        from_attributes = True

class SessionResponse(BaseModel):
    session_id: int
    user_id: int

    class Config:
        from_attributes = True

T = TypeVar("T")

class PaginatedResponse(GenericModel, Generic[T]):
    items: List[T]
    next_cursor: Optional[int] = None