import os

# VLLM API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# База данных
DATABASE_URL = os.getenv("DATABASE_URL")

# JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# LLM
MODEL = os.getenv("MODEL")
_LORA_MODULES_RAW = os.getenv("LORA_MODULES", "")
LORA_MODULES = {}
for item in _LORA_MODULES_RAW.split(","):
    if "=" in item:
        name, path = item.split("=", 1)
        LORA_MODULES[name.strip()] = path.strip()
