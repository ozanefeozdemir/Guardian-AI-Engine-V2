from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from database import Base
import datetime

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id', ondelete="CASCADE"), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id', ondelete="CASCADE"), primary_key=True)
)

class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String) 

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    permissions = relationship("Permission", secondary=role_permissions, lazy="joined")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="SET NULL"), nullable=True)
    role = relationship("Role", lazy="joined")
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

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

class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True)
    # Hatalı girişlerde (olmayan kullanıcı adıyla) patlamamak için String tutuyoruz:
    username = Column(String, index=True, nullable=False) 
    ip_address = Column(String, nullable=False)
    action = Column(String, nullable=False) # Örn: "Başarılı Giriş", "Yanlış Şifre", "Kullanıcı Yok"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)