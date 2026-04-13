import os
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from passlib.context import CryptContext

# 1. .env dosyasını yükle
load_dotenv()

# 2. Database Bağlantısı (IPv4 kullanarak localhost karmaşasını önlüyoruz)
DATABASE_URL = "postgresql+asyncpg://guardian_user:guardian_pass@127.0.0.1:5432/guardian_db"
print(f"\n🔍 DEBUG: Python'ın okuduğu URL -> {DATABASE_URL}\n")

DEFAULT_ADMIN_USER = os.getenv("DEFAULT_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS = os.getenv("DEFAULT_ADMIN_PASS", "admin123")

# 3. Şifreleme motoru
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 4. Engine ve Session ayarları
engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# 5. Database Fonksiyonları
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def seed_data():
    """Veritabanı boşsa varsayılan admin kullanıcısını oluşturur."""
    from models import User
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        user = result.scalars().first()

        if not user:
            print("🚀 Veritabanı boş! Default admin oluşturuluyor...")
            
            hashed_pw = pwd_context.hash(DEFAULT_ADMIN_PASS)
            
            # NOT NULL hatasını önlemek için email ve role eklendi
            new_user = User(
                username=DEFAULT_ADMIN_USER,
                email=f"{DEFAULT_ADMIN_USER}@guardian.ai", # Bu satır hatayı çözer
                hashed_password=hashed_pw,
                role="admin",
                is_active=True
            )
            
            session.add(new_user)
            await session.commit()
            print(f"✅ Başarıyla oluşturuldu: {DEFAULT_ADMIN_USER}")
        else:
            print("ℹ️ Veritabanında kullanıcı mevcut, seed işlemi atlandı.")