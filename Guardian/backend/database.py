from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Veritabanı Bağlantı Adresi (URL)
# Bu adres, docker-compose.yml ve .env dosyasında belirttiğimiz bilgilerle eşleşmelidir.
# Format: "postgresql://KULLANICI_ADI:SIFRE@HOST_ADI:PORT/VERITABANI_ADI"
#
# ÖNEMLİ: Şu anda FastAPI uygulamamız kendi bilgisayarımızda (Docker dışında) çalıştığı için,
# Docker'ın dışarıya açtığı porta "localhost" üzerinden bağlanıyoruz.
# Uygulamayı Docker'a koyduğumuzda burası "db" olarak değişecek.
SQLALCHEMY_DATABASE_URL = "postgresql://guardian_user:guardian_password@localhost:5432/guardian_db"

# SQLAlchemy motorunu oluşturuyoruz. Bu, veritabanıyla olan temel bağlantı noktasıdır.
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Veritabanıyla "konuşmalar" (session) yapmak için bir fabrika oluşturuyoruz.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Modellerimizin (tablo yapılarımızın) miras alacağı temel bir sınıf oluşturuyoruz.
Base = declarative_base()