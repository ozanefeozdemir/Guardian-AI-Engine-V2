import os
import warnings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

_default_db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/guardian_ai"
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = _default_db_url
    warnings.warn(
        "⚠️  DATABASE_URL ortam değişkeni ayarlanmadı! "
        "Varsayılan yerel bağlantı kullanılıyor. "
        "Üretimde güçlü bir şifre ile DATABASE_URL ayarlayın.",
        stacklevel=1,
    )

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
