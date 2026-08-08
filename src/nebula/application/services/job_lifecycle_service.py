"""Database-backed leases and recovery for persisted background jobs."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from sqlalchemy import or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import PipelineRun, SyncTask
from nebula.db.database import get_db_context
from nebula.domain import PipelineStatus
from nebula.utils import get_logger

logger = get_logger(__name__)

JobKind = Literal["pipeline", "sync_task"]
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

LEASE_DURATION = timedelta(minutes=5)
HEARTBEAT_INTERVAL_SECONDS = 30.0
HEARTBEAT_RETRY_SECONDS = 5.0
ACTIVE_STATUSES = (PipelineStatus.pending.value, PipelineStatus.running.value)
WORKER_PREFIX = f"{socket.gethostname()}:{os.getpid()}"


class JobLeaseLostError(RuntimeError):
    """Raised when an orchestration no longer owns its persisted lease."""


def utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def apply_job_lease(
    record: PipelineRun | SyncTask,
    *,
    now: datetime | None = None,
    worker_id: str | None = None,
) -> str:
    """Attach a unique execution lease to a newly-created job record."""
    heartbeat_at = now or utcnow()
    execution_token = worker_id or f"{WORKER_PREFIX}:{uuid4().hex}"
    record.heartbeat_at = heartbeat_at
    record.lease_expires_at = heartbeat_at + LEASE_DURATION
    record.worker_id = execution_token
    return execution_token


def clear_job_lease(record: PipelineRun | SyncTask) -> None:
    """Clear ownership when a job reaches a terminal state."""
    record.lease_expires_at = None
    record.worker_id = None


def active_lease_condition(model: type[PipelineRun] | type[SyncTask]):
    """Build the canonical SQL predicate for a currently active leased job."""
    return (
        model.status.in_(ACTIVE_STATUSES),
        model.lease_expires_at.is_not(None),
        model.lease_expires_at > utcnow(),
    )


def _model_for(kind: JobKind) -> type[PipelineRun] | type[SyncTask]:
    return PipelineRun if kind == "pipeline" else SyncTask


async def renew_job_lease(
    kind: JobKind,
    record_id: int,
    worker_id: str,
    *,
    session_factory: SessionFactory = get_db_context,
) -> bool:
    """Renew only a still-current, unexpired execution lease."""
    model = _model_for(kind)
    now = utcnow()
    async with session_factory() as db:
        result = await db.execute(
            update(model)
            .where(
                model.id == record_id,
                model.status.in_(ACTIVE_STATUSES),
                model.worker_id == worker_id,
                model.lease_expires_at.is_not(None),
                model.lease_expires_at > now,
            )
            .values(
                heartbeat_at=now,
                lease_expires_at=now + LEASE_DURATION,
            )
        )
        await db.commit()
        return bool(result.rowcount)


async def interrupt_expired_job(
    db: AsyncSession,
    kind: JobKind,
    record_id: int,
    *,
    user_id: int | None = None,
) -> bool:
    """Atomically make one expired active job visible as interrupted."""
    model = _model_for(kind)
    now = utcnow()
    conditions = [
        model.id == record_id,
        model.status.in_(ACTIVE_STATUSES),
        or_(model.lease_expires_at.is_(None), model.lease_expires_at <= now),
    ]
    if user_id is not None:
        conditions.append(model.user_id == user_id)
    values: dict[str, object] = {
        "status": PipelineStatus.interrupted.value,
        "completed_at": now,
        "lease_expires_at": None,
        "worker_id": None,
    }
    if kind == "pipeline":
        values.update(
            phase=PipelineStatus.interrupted.value,
            last_error="Job interrupted because its worker lease expired",
        )
    else:
        values.update(
            error_message="Job interrupted because its worker lease expired",
        )
    result = await db.execute(update(model).where(*conditions).values(**values))
    return bool(result.rowcount)


async def interrupt_expired_jobs_for_user(
    db: AsyncSession,
    user_id: int,
) -> tuple[int, int]:
    """Reconcile all expired active owners before launching user work."""
    now = utcnow()
    reason = "Job interrupted because its worker lease expired"
    runs = await db.execute(
        update(PipelineRun)
        .where(
            PipelineRun.user_id == user_id,
            PipelineRun.status.in_(ACTIVE_STATUSES),
            or_(
                PipelineRun.lease_expires_at.is_(None),
                PipelineRun.lease_expires_at <= now,
            ),
        )
        .values(
            status=PipelineStatus.interrupted.value,
            phase=PipelineStatus.interrupted.value,
            last_error=reason,
            completed_at=now,
            lease_expires_at=None,
            worker_id=None,
        )
    )
    tasks = await db.execute(
        update(SyncTask)
        .where(
            SyncTask.user_id == user_id,
            SyncTask.task_type == "full_refresh",
            SyncTask.status.in_(ACTIVE_STATUSES),
            or_(
                SyncTask.lease_expires_at.is_(None),
                SyncTask.lease_expires_at <= now,
            ),
        )
        .values(
            status=PipelineStatus.interrupted.value,
            error_message=reason,
            completed_at=now,
            lease_expires_at=None,
            worker_id=None,
        )
    )
    return int(runs.rowcount or 0), int(tasks.rowcount or 0)


class JobLeaseHeartbeat:
    """Keep one persisted job lease alive for the lifetime of orchestration."""

    def __init__(
        self,
        kind: JobKind,
        record_id: int,
        worker_id: str,
        *,
        session_factory: SessionFactory = get_db_context,
    ) -> None:
        self.kind = kind
        self.record_id = record_id
        self.worker_id = worker_id
        self.session_factory = session_factory
        self._task: asyncio.Task[None] | None = None
        self._owner_task: asyncio.Task[object] | None = None
        self._lease_deadline: datetime | None = None
        self._lost = False

    async def start(self) -> None:
        renewed = await renew_job_lease(
            self.kind,
            self.record_id,
            self.worker_id,
            session_factory=self.session_factory,
        )
        if not renewed:
            raise JobLeaseLostError(
                f"Cannot start {self.kind} {self.record_id}: execution lease is not owned"
            )
        self._lease_deadline = utcnow() + LEASE_DURATION
        self._owner_task = asyncio.current_task()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive task boundary
            logger.warning(
                "Lease heartbeat stopped after an unexpected error "
                f"kind={self.kind} record_id={self.record_id}: {exc}"
            )
        self._task = None

    def ensure_owned(self) -> None:
        """Fail a phase boundary after the execution lease has been lost."""
        if self._lost:
            raise JobLeaseLostError(
                f"Lost {self.kind} execution lease for record {self.record_id}"
            )

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            try:
                renewed = await renew_job_lease(
                    self.kind,
                    self.record_id,
                    self.worker_id,
                    session_factory=self.session_factory,
                )
            except Exception as exc:
                logger.warning(
                    "Lease heartbeat renewal failed; retrying "
                    f"kind={self.kind} record_id={self.record_id}: {exc}"
                )
                if self._lease_deadline is not None and utcnow() < self._lease_deadline:
                    await asyncio.sleep(HEARTBEAT_RETRY_SECONDS)
                    continue
                await self._mark_lost()
                return
            if not renewed:
                await self._mark_lost()
                return
            self._lease_deadline = utcnow() + LEASE_DURATION

    async def _mark_lost(self) -> None:
        self._lost = True
        try:
            async with self.session_factory() as db:
                await interrupt_expired_job(db, self.kind, self.record_id)
                await db.commit()
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.warning(
                "Failed to persist lost execution lease "
                f"kind={self.kind} record_id={self.record_id}: {exc}"
            )
        if self._owner_task is not None and not self._owner_task.done():
            self._owner_task.cancel()


async def reconcile_orphaned_jobs(
    *, session_factory: SessionFactory = get_db_context
) -> tuple[int, int]:
    """Mark expired or pre-lease active jobs interrupted at application startup."""
    now = utcnow()
    reason = "Job interrupted because its worker lease expired"
    stale_lease = or_(
        PipelineRun.lease_expires_at.is_(None),
        PipelineRun.lease_expires_at <= now,
    )
    stale_task_lease = or_(
        SyncTask.lease_expires_at.is_(None),
        SyncTask.lease_expires_at <= now,
    )
    async with session_factory() as db:
        runs = await db.execute(
            update(PipelineRun)
            .where(PipelineRun.status.in_(ACTIVE_STATUSES), stale_lease)
            .values(
                status=PipelineStatus.interrupted.value,
                phase=PipelineStatus.interrupted.value,
                last_error=reason,
                completed_at=now,
                lease_expires_at=None,
                worker_id=None,
            )
        )
        tasks = await db.execute(
            update(SyncTask)
            .where(SyncTask.status.in_(ACTIVE_STATUSES), stale_task_lease)
            .values(
                status=PipelineStatus.interrupted.value,
                error_message=reason,
                completed_at=now,
                lease_expires_at=None,
                worker_id=None,
            )
        )
        await db.commit()
    run_count = int(runs.rowcount or 0)
    task_count = int(tasks.rowcount or 0)
    if run_count or task_count:
        logger.warning(
            "Recovered orphaned background jobs "
            f"pipeline_runs={run_count} sync_tasks={task_count}"
        )
    return run_count, task_count
