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
