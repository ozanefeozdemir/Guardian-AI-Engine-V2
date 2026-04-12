from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from database import Base
import datetime

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, nullable=False) # Unix timestamp
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String, index=True)
    is_attack = Column(Boolean, default=False)
    confidence = Column(Float)
    attack_type = Column(String, index=True)
    original_features = Column(JSON)

# --- YENİ EKLENENLER: KİMLİK DOĞRULAMA VE LOG TABLOLARI ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="admin") # Herkes giremesin diye default admin yapıyoruz
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True)
    # Hatalı girişlerde (olmayan kullanıcı adıyla) patlamamak için String tutuyoruz:
    username = Column(String, index=True, nullable=False) 
    ip_address = Column(String, nullable=False)
    action = Column(String, nullable=False) # Örn: "Başarılı Giriş", "Yanlış Şifre", "Kullanıcı Yok"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)