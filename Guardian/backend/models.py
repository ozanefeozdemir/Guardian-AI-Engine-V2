from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base  

class Alert(Base):
    # Bu modelin veritabanında hangi tabloya karşılık geleceğini belirtiyoruz.
    __tablename__ = "alerts"

    # Tablonun sütunlarını ve özelliklerini tanımlıyoruz.
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    reconstruction_error = Column(Float)