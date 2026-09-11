"""Embedding execution for the sync pipeline."""

from datetime import datetime, timezone
from time import perf_counter

from sqlalchemy import func, select

from nebula.db import StarredRepo, SyncTask
from nebula.utils import get_logger

from .sync_execution_support import (
    generate_repo_enhancements_in_parallel,
    log_task_stage,
)

logger = get_logger(__name__)


def _facade():
    from nebula.application.services import sync_execution_service as module

    return module


async def _embed_one_chunk(
    *,
    repos: list[StarredRepo],
    llm_service,
    embedding_service,
    sync_settings,
) -> int:
    """Enhance, embed, and persist one chunk. Returns the LLM failure count.

    Raises on embedding failure so the caller can account for the whole chunk.
    """
    llm_failed = 0
    repos_needing_llm = [
        repo for repo in repos if not repo.ai_summary or not repo.ai_tags
    ]

    if repos_needing_llm:
        llm_results = await generate_repo_enhancements_in_parallel(
            llm_service,
            repos_needing_llm,
            concurrency=sync_settings.llm_enhancement_concurrency,
        )
        for repo in repos_needing_llm:
            summary, tags, error = llm_results.get(
                repo.id, (None, None, RuntimeError("Missing LLM result"))
            )
            if error is None:
                repo.ai_summary = summary
                repo.ai_tags = tags
                repo.is_summarized = True
            else:
                llm_failed += 1
                logger.warning(f"LLM generation failed for {repo.full_name}: {error}")
                if not repo.ai_tags:
                    repo.ai_tags = repo.topics[:5] if repo.topics else ["开源项目"]

    texts = []
    for repo in repos:
        text = embedding_service.build_repo_text(
            full_name=repo.full_name,
            description=repo.description,
            topics=repo.topics,
            readme_content=repo.readme_content,
            language=repo.language,
            ai_summary=repo.ai_summary,
            ai_tags=repo.ai_tags,
        )
        texts.append(text)
        repo.embedding_text = text

    embeddings = await embedding_service.embed_batch(
        texts,
        batch_size=sync_settings.batch_size,
    )
    if len(embeddings) != len(repos):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} for {len(repos)} repos"
        )

    for repo, embedding in zip(repos, embeddings, strict=True):
        repo.embedding = embedding
        repo.is_embedded = True

    return llm_failed


async def compute_embeddings_task(user_id: int, task_id: int):
    """Background task to compute embeddings for repos.

    Repos are processed in `SYNC_BATCH_SIZE` chunks, each committed before the
    next starts, so `is_embedded` acts as a durable resume marker. A failing
    chunk is counted and skipped rather than discarding the whole run: with a
    large star collection, an all-or-nothing pass throws away paid embedding
    work on any single transient fault.

    Chunk iteration uses a keyset cursor on `StarredRepo.id` and advances the
    cursor before processing, so a chunk that keeps failing cannot loop forever.
    It stays `is_embedded = False` and is retried by the next task run.
    """
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            task = await db.get(SyncTask, task_id)
            if not task:
                return

            task.status = "running"
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

            total_result = await db.execute(
                select(func.count(StarredRepo.id)).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.is_embedded == False,  # noqa: E712
                )
            )
            total_pending = int(total_result.scalar() or 0)
            task.total_items = total_pending
            await db.commit()

            if total_pending == 0:
                task.status = "completed"
                task.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            sync_settings = _facade().get_sync_settings()
            llm_service = _facade().get_llm_service()
            embedding_service = _facade().get_embedding_service()
            chunk_size = sync_settings.batch_size

            processed = 0
            failed = 0
            llm_failed_total = 0
            chunks_succeeded = 0
            chunks_failed = 0
            last_id = 0

            while True:
                chunk_result = await db.execute(
                    select(StarredRepo)
                    .where(
                        StarredRepo.user_id == user_id,
                        StarredRepo.is_embedded == False,  # noqa: E712
                        StarredRepo.id > last_id,
                    )
                    .order_by(StarredRepo.id)
                    .limit(chunk_size)
                )
                chunk = list(chunk_result.scalars().all())
                if not chunk:
                    break

                chunk_first_id = chunk[0].id
                chunk_last_id = chunk[-1].id
                # Advance before processing: a chunk that keeps failing must not
                # be re-selected on this pass.
                last_id = chunk_last_id

                chunk_started = perf_counter()
                try:
                    llm_failed_total += await _facade()._embed_one_chunk(
                        repos=chunk,
                        llm_service=llm_service,
                        embedding_service=embedding_service,
                        sync_settings=sync_settings,
                    )
                    processed += len(chunk)
                    task.processed_items = processed
                    task.failed_items = failed
                    await db.commit()
                    chunks_succeeded += 1
                    log_task_stage(
                        "compute_embeddings_task",
                        "embed_chunk",
                        chunk_started,
                        repos=len(chunk),
                        first_id=chunk_first_id,
                        last_id=chunk_last_id,
                        embedded_total=processed,
                    )
                except Exception as exc:
                    await db.rollback()
                    failed += len(chunk)
                    chunks_failed += 1
                    logger.exception(
                        "Embedding chunk failed "
                        f"user_id={user_id} task_id={task_id} "
                        f"ids={chunk_first_id}..{chunk_last_id} "
                        f"size={len(chunk)}: {exc}"
                    )
                    task = await db.get(SyncTask, task_id)
                    if task is None:
                        return
                    task.processed_items = processed
                    task.failed_items = failed
                    await db.commit()

            task = await db.get(SyncTask, task_id)
            if task is None:
                return

            task.processed_items = processed
            task.failed_items = failed
            task.error_details = {
                "llm_failed_items": llm_failed_total,
                "embedded_items": processed,
                "chunks_succeeded": chunks_succeeded,
                "chunks_failed": chunks_failed,
                "chunk_size": chunk_size,
            }
            if chunks_succeeded == 0 and chunks_failed > 0:
                # Nothing landed at all: keep today's hard-failure behaviour so
                # the pipeline stops instead of building a snapshot on no data.
                task.status = "failed"
                task.error_message = (
                    f"All {chunks_failed} embedding chunks failed for user {user_id}"
                )
            else:
                task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(
                f"Completed embedding for user {user_id}: {processed} embedded, "
                f"{failed} failed, chunks ok={chunks_succeeded} failed={chunks_failed}"
            )
        except Exception as exc:
            logger.exception(f"Embedding task failed: {exc}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(exc)
                    task.completed_at = datetime.now(timezone.utc)
                    await db.commit()

