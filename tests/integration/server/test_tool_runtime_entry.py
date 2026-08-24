from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lib.episode_reset import EpisodeResetResult
from lib.project_manager import ProjectManager
from lib.project_migration_failure import MIGRATION_FAILURE_CODE, record_migration_failure
from server.tool_runtime import (
    CallerContext,
    CreateProjectToolRequest,
    PatchProjectRequest,
    ProjectScope,
    RenameAssetRequest,
    ResetEpisodePlanningRequest,
    Services,
    ToolRequest,
    UploadSourceRequest,
    create_project,
    get_source_text,
    list_project_files,
    list_projects,
    list_source_files,
    patch_project,
    rename_asset,
    reset_episode_planning,
    upload_source,
)


class _Unused:
    pass


def _services(tmp_path: Path) -> Services:
    return Services(
        projects=ProjectManager(tmp_path / "projects"),
        workflow_planner=_Unused(),  # type: ignore[arg-type]
        capabilities=_Unused(),  # type: ignore[arg-type]
    )


async def test_entry_handlers_create_list_and_upload_a_readable_source(tmp_path: Path) -> None:
    services = _services(tmp_path)
    caller = CallerContext(user_id="test", source="mcp")

    created = await create_project(
        ToolRequest(
            CreateProjectToolRequest(
                name="demo",
                title="Demo",
                content_mode="narration",
                generation_mode="storyboard",
            )
        ),
        caller,
        services,
    )
    projects = await list_projects(ToolRequest(None), caller, services)
    scope = ProjectScope(project_name="demo", projects_root=services.projects.projects_root)
    uploaded = await upload_source(
        ToolRequest(UploadSourceRequest(filename="novel.txt", content="第一章\n你好")),
        scope,
        caller,
        services,
    )
    source_files = await list_source_files(ToolRequest(None), scope, caller, services)
    source_text = await get_source_text(ToolRequest("source/novel.txt"), scope, caller, services)

    assert created.problem is None
    assert created.value is not None
    assert created.value["name"] == "demo"
    assert projects.value == [
        {
            "name": "demo",
            "title": "Demo",
            "content_mode": "narration",
            "generation_mode": "storyboard",
        }
    ]
    assert uploaded.value == {
        "filename": "novel.txt",
        "path": "source/novel.txt",
        "original_filename": "novel.txt",
        "original_kept": False,
        "used_encoding": "utf-8",
        "chapter_count": 0,
    }
    assert source_files.value is not None
    assert [entry.path for entry in source_files.value.files] == ["source/novel.txt"]
    assert source_text.value is not None
    assert source_text.value.path == "source/novel.txt"
    assert source_text.value.text == "第一章\n你好"


async def test_list_projects_uses_readonly_loader(tmp_path: Path, monkeypatch) -> None:
    services = _services(tmp_path)
    services.projects.create_project("demo")
    services.projects.create_project_metadata("demo", "Demo")

    def mutating_loader(_name: str):
        raise AssertionError("list_projects must not run migration writes")

    monkeypatch.setattr(services.projects, "load_project", mutating_loader)

    outcome = await list_projects(
        ToolRequest(None),
        CallerContext(user_id="test", source="mcp"),
        services,
    )

    assert outcome.problem is None
    assert outcome.value and outcome.value[0]["name"] == "demo"


async def test_entry_handlers_return_typed_problems(tmp_path: Path) -> None:
    services = _services(tmp_path)
    caller = CallerContext(user_id="test", source="embedded")

    missing = await upload_source(
        ToolRequest(UploadSourceRequest(filename="novel.txt", content="hello")),
        ProjectScope(project_name="missing", projects_root=services.projects.projects_root),
        caller,
        services,
    )

    assert missing.problem is not None
    assert missing.problem.code == "project_not_found"


@pytest.mark.parametrize(
    ("filename", "expected_code"),
    [
        ("../novel.txt", "invalid_request"),
        (".secret.txt", "invalid_request"),
        ("novel.pdf", "unsupported_format"),
    ],
)
async def test_upload_source_rejects_unsafe_or_non_text_filenames(
    tmp_path: Path, filename: str, expected_code: str
) -> None:
    services = _services(tmp_path)
    services.projects.create_project("demo")
    services.projects.create_project_metadata("demo", "Demo")
    caller = CallerContext(user_id="test", source="mcp")

    outcome = await upload_source(
        ToolRequest(UploadSourceRequest(filename=filename, content="hello")),
        ProjectScope(project_name="demo", projects_root=services.projects.projects_root),
        caller,
        services,
    )

    assert outcome.problem is not None
    assert outcome.problem.code == expected_code


async def test_upload_source_rejects_symlinked_source_directory(tmp_path: Path) -> None:
    services = _services(tmp_path)
    project_dir = services.projects.create_project("demo")
    services.projects.create_project_metadata("demo", "Demo")
    source_dir = project_dir / "source"
    source_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source_dir.symlink_to(outside, target_is_directory=True)

    outcome = await upload_source(
        ToolRequest(UploadSourceRequest(filename="novel.txt", content="hello")),
        ProjectScope(project_name="demo", projects_root=services.projects.projects_root),
        CallerContext(user_id="test", source="mcp"),
        services,
    )

    assert outcome.problem is not None
    assert outcome.problem.code == "invalid_request"
    assert not (outside / "novel.txt").exists()


async def test_upload_source_respects_migration_failure_gate(tmp_path: Path) -> None:
    services = _services(tmp_path)
    project_dir = services.projects.create_project("demo")
    record_migration_failure(project_dir, ValueError("repair required"), schema_version=7)

    outcome = await upload_source(
        ToolRequest(UploadSourceRequest(filename="novel.txt", content="hello")),
        ProjectScope(project_name="demo", projects_root=services.projects.projects_root),
        CallerContext(user_id="test", source="mcp"),
        services,
    )

    assert outcome.problem is not None
    assert outcome.problem.code == MIGRATION_FAILURE_CODE
    assert not (project_dir / "source" / "novel.txt").exists()


@pytest.mark.parametrize("handler", [list_source_files, list_project_files])
async def test_file_enumeration_runs_off_event_loop(tmp_path: Path, monkeypatch, handler) -> None:
    from server import tool_runtime as mod

    services = _services(tmp_path)
    services.projects.create_project("demo")
    offloaded: list[str] = []

    async def run_in_thread(function, /, *args, **kwargs):
        offloaded.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(mod.asyncio, "to_thread", run_in_thread)
    outcome = await handler(
        ToolRequest(None),
        ProjectScope(project_name="demo", projects_root=services.projects.projects_root),
        CallerContext(user_id="test", source="mcp"),
        services,
    )

    assert outcome.problem is None
    assert offloaded == ["_business_file_entries"]


async def test_filesystem_heavy_mutations_run_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    services = _services(tmp_path)
    services.projects.create_project("demo")
    services.projects.create_project_metadata("demo", "Demo")
    services.projects.upsert_assets("demo", "characters", {"旧名": {"description": "角色"}})
    scope = ProjectScope(project_name="demo", projects_root=services.projects.projects_root)
    caller = CallerContext(user_id="test", source="mcp")
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []

    def resetter(_project_path: Path, *, from_episode: int, confirm_consumed: bool) -> EpisodeResetResult:
        worker_threads.append(threading.get_ident())
        return EpisodeResetResult([], [], [], [])

    reset = await reset_episode_planning(
        ToolRequest(ResetEpisodePlanningRequest(from_episode=1)),
        scope,
        caller,
        services,
        resetter=resetter,
    )

    original_rename = services.projects.rename_asset

    def tracked_rename(project_name: str, table: str, old_name: str, new_name: str):
        worker_threads.append(threading.get_ident())
        return original_rename(project_name, table, old_name, new_name)

    monkeypatch.setattr(services.projects, "rename_asset", tracked_rename)
    renamed = await rename_asset(
        ToolRequest(RenameAssetRequest(table="characters", old_name="旧名", new_name="新名")),
        scope,
        caller,
        services,
    )

    original_upsert = services.projects.upsert_assets

    def tracked_upsert(project_name: str, table: str, entries: dict):
        worker_threads.append(threading.get_ident())
        return original_upsert(project_name, table, entries)

    monkeypatch.setattr(services.projects, "upsert_assets", tracked_upsert)
    patched = await patch_project(
        ToolRequest(PatchProjectRequest(table="characters", entries={"另一角色": {"description": "角色"}})),
        scope,
        caller,
        services,
    )

    assert reset.problem is None
    assert renamed.problem is None
    assert patched.problem is None
    assert len(worker_threads) == 3
    assert all(thread != caller_thread for thread in worker_threads)


def test_create_project_request_rejects_mode_specific_fields_before_writing(tmp_path: Path) -> None:
    services = _services(tmp_path)

    with pytest.raises(ValueError):
        CreateProjectToolRequest(name="demo", content_mode="narration", target_duration=30)

    assert services.projects.list_projects() == []
