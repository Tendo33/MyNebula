"""Job entrypoint for snapshot rebuild."""

from nebula.application.services import GraphQueryService
from nebula.application.services.user_service import get_default_user
from nebula.db.database import get_db_context


async def run_snapshot_build_job() -> None:
    """Build and activate a new graph snapshot."""
    service = GraphQueryService()
    async with get_db_context() as db:
        user = await get_default_user(db)
        await service.rebuild_active_snapshot(db, user=user)
