import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from nebula.core.config import get_database_settings
from nebula.db.models import Base


async def reset_database():
    """Drop all tables in the database."""
    settings = get_database_settings()

    print(f"Connecting to database: {settings.host}:{settings.port}/{settings.name}")
    print("WARNING: This will DROP ALL TABLES. Ctrl+C to cancel in 5 seconds...")
    await asyncio.sleep(5)

    engine = create_async_engine(settings.async_url, echo=True)

    async with engine.begin() as conn:
        print("Dropping all tables...")
        # Reflect and drop is safer, but Base.metadata.drop_all assumes we know all tables.
        # Since we just want to clear *our* tables, Base.metadata.drop_all is fine.
        # But if there are migration tables (alembic_version), drop_all might not drop them if not in Base?
        # Alembic table is usually not in Base.

        # We should also drop alembic_version manually to reset migration history.
        await conn.run_sync(Base.metadata.drop_all)
        print("Application tables dropped.")

        # Drop alembic_version table
        print("Dropping alembic_version table...")
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        print("alembic_version dropped.")

    await engine.dispose()
    print("Database reset complete.")


if __name__ == "__main__":
    try:
        asyncio.run(reset_database())
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
