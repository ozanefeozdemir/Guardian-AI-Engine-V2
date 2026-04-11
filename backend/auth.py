from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel

from database import get_db
from models import User, AuthLog

# --- Güvenlik Ayarları ---
SECRET_KEY = "guardian-super-secret-key-donot-share"  # JWT şifreleme anahtarımız
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token 1 saat geçerli olacak

# Şifre hashleme algoritması
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter()

# --- Pydantic Şemaları (Gelen veriyi doğrulamak için) ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# --- Yardımcı Fonksiyonlar ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def log_auth_action(db: AsyncSession, username: str, ip: str, action: str):
    """Sisteme giriş denemelerini IP adresiyle fişler."""
    new_log = AuthLog(username=username, ip_address=ip, action=action)
    db.add(new_log)
    await db.commit()

# --- ENDPOINTLER (API Kapıları) ---

@router.post("/register")
async def register_user(user: UserCreate, request: Request, db: AsyncSession = Depends(get_db)):
    # 1. Kullanıcı adı alınmış mı kontrol et
    stmt = select(User).where(User.username == user.username)
    result = await db.execute(stmt)
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış")
    
    # 2. Şifreyi hashle ve kaydet
    hashed_pw = get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    await db.commit()
    
    # 3. Logla
    client_ip = request.client.host if request.client else "Bilinmiyor"
    await log_auth_action(db, user.username, client_ip, "Yeni Kullanıcı Kaydı")
    
    return {"message": "Kayıt başarıyla oluşturuldu!"}

@router.post("/login")
async def login_user(user_data: UserLogin, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else "Bilinmiyor"
    
    # 1. Kullanıcıyı bul
    stmt = select(User).where(User.username == user_data.username)
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    # 2. Kullanıcı yoksa veya şifre yanlışsa
    if not user or not verify_password(user_data.password, user.hashed_password):
        await log_auth_action(db, user_data.username, client_ip, "BAŞARISIZ GİRİŞ - Yanlış Şifre/Kullanıcı")
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı adı veya şifre")
        
    # 3. Giriş Başarılı
    await log_auth_action(db, user_data.username, client_ip, "Başarılı Giriş")
    
    # 4. Token (Yaka Kartı) Üret ve Ver
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username, "role": user.role}

@router.get("/logs")
async def get_auth_logs(limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Dashboard'da göstermek için giriş-çıkış loglarını getirir."""
    stmt = select(AuthLog).order_by(AuthLog.id.desc()).limit(limit)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return logs