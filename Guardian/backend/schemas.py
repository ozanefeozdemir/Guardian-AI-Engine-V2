from pydantic import BaseModel
import datetime

# --- Alert (Uyarı) için Şemalar ---

# Bir uyarının temel, ortak alanlarını içeren şema.
# Diğer şemalarımız bu temel şemayı miras alacak.
class AlertBase(BaseModel):
    ip_address: str
    reconstruction_error: float

# Yeni bir uyarı OLUŞTURURKEN kullanılacak şema.
# API'ye bu formatta veri gönderilmesini bekleyeceğiz.
class AlertCreate(AlertBase):
    pass  # Şimdilik AlertBase ile aynı, ekstra bir alana ihtiyacımız yok.

# Veritabanından bir uyarı OKURKEN veya oluşturduktan sonra
# API'den bu formatta veri DÖNDÜRECEĞİZ.
class Alert(AlertBase):
    id: int
    timestamp: datetime.datetime

    # Bu ayar, Pydantic'e "Sana bir SQLAlchemy objesi geldiğinde,
    # onu okuyup JSON'a çevirebilirsin" demenin yoludur.
    class Config:
        orm_mode = True