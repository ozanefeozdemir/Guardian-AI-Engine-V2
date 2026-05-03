import os
import warnings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
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
    await _seed_rbac()


# Permissions used by the IP whitelist/blacklist endpoints. Kept here so the
# DB always boots into a consistent state — admin role gets both on every boot.
_SEED_PERMISSIONS = [
    ("view_ip_rules",   "Read IP whitelist/blacklist rules"),
    ("manage_ip_rules", "Create / update / delete IP whitelist/blacklist rules"),
]


async def _seed_rbac():
    """Idempotently create core permissions and attach them to the admin role."""
    from sqlalchemy.orm import selectinload
    from models import Permission, Role

    async with AsyncSessionLocal() as session:
        perm_objs = {}
        for name, desc in _SEED_PERMISSIONS:
            existing = (
                await session.execute(select(Permission).where(Permission.name == name))
            ).scalars().first()
            if not existing:
                existing = Permission(name=name, description=desc)
                session.add(existing)
            perm_objs[name] = existing
        await session.flush()

        # Eager-load `permissions` so iterating it doesn't trigger sync lazy IO
        # under the async session.
        admin = (
            await session.execute(
                select(Role).options(selectinload(Role.permissions)).where(Role.name == "admin")
            )
        ).scalars().first()
        if not admin:
            admin = Role(name="admin", permissions=list(perm_objs.values()))
            session.add(admin)
        else:
            existing_names = {p.name for p in admin.permissions}
            for name, perm in perm_objs.items():
                if name not in existing_names:
                    admin.permissions.append(perm)

        await session.commit()
