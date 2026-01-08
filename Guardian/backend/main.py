from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

import models
import schemas  # Artık schemas.py dosyamızı da tanıyoruz
from database import engine, SessionLocal # SessionLocal'i de import ediyoruz

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Guardian Backend API")

def get_db():
    db = SessionLocal()  
    try:
        yield db  
    finally:
        db.close() 


@app.get("/")
def read_root():
    return {"message": "Guardian API is running and connected!"}


@app.get("/alerts/", response_model=list[schemas.Alert])
def read_alerts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Veritabanındaki tüm uyarıları, sayfalama (skip, limit) yaparak döndürür.
    """
    # db.query(models.Alert) -> "Alert" tablosuna bir sorgu hazırla.
    # .offset(skip).limit(limit) -> Sayfalama uygula (ilk 'skip' taneyi atla, 'limit' kadar al).
    # .all() -> Sorguyu çalıştır ve tüm sonuçları bir liste olarak getir.
    alerts = db.query(models.Alert).offset(skip).limit(limit).all()
    return alerts

@app.post("/alerts/", response_model=schemas.Alert)
def create_alert(alert: schemas.AlertCreate, db: Session = Depends(get_db)):
    """
    Verilen 'alert' verisine göre yeni bir uyarı kaydı oluşturur.
    """
    # 1. Gelen JSON verisini (alert) bir veritabanı objesine (models.Alert) dönüştür.
    db_alert = models.Alert(
        ip_address=alert.ip_address, 
        reconstruction_error=alert.reconstruction_error
    )
    
    # 2. Hazırlanan objeyi "görüşme odasına" (session) ekle. (Henüz veritabanına kaydetmedi)
    db.add(db_alert)
    
    # 3. Değişiklikleri veritabanına kalıcı olarak işle (INSERT komutu burada çalışır).
    db.commit()
    
    # 4. Veritabanının oluşturduğu 'id' ve 'timestamp' gibi yeni bilgileri
    #    'db_alert' objemize geri yükle.
    db.refresh(db_alert)
    
    # 5. Oluşturulan kaydın son halini (şemaya uygun olarak) JSON formatında döndür.
    return db_alert