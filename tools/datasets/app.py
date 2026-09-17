"""Serve the task-plan, object-asset, and collection QC browser.

Plans are discovered as batches below task_sets. Objects live in one shared
asset catalog, while collected episodes remain in their existing LeRobot
directories and are joined to plans through stable slot IDs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import zipfile
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import Forbidden, HTTPException

from tools.datasets.collection import PlanCatalog
from tools.datasets.dataset_transfer import DatasetTransfer
from tools.datasets.hf_task_sets import (
    fetch_dataset,
    fetch_qc,
    fetch_task_set,
    publish_assets,
    publish_qc,
    publish_task_sets,
)
from tools.datasets.store import ConflictError, RecordNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets/data_collection"
DEFAULT_PLANS_ROOT = DEFAULT_DATA_ROOT / "task_sets"
DEFAULT_ASSETS_ROOT = DEFAULT_DATA_ROOT / "assets"
DEFAULT_COLLECTION_ROOT = DEFAULT_DATA_ROOT
CONSOLE_STATIC_ROOT = PROJECT_ROOT / "src/core/app/console/static"


class DatasetService:
    """Own the dataset catalog and its HTTP endpoints."""

    def __init__(self, plans_root, assets_root, collection_root, read_only, locale="en"):
        self.app = app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
        # The editor is frequently updated during collection; avoid browsers
        # retaining stale module code after a service restart.
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
        app.config["DATASET_READ_ONLY"] = read_only
        app.config["DATASET_LOCALE"] = locale
        self.catalog = PlanCatalog(plans_root, assets_root, collection_root)
        self.transfers = DatasetTransfer(PROJECT_ROOT, self.catalog)

        # Register the catalog and review endpoints
        app.after_request(self.compress_json)
        app.after_request(self.disable_static_cache)
        app.before_request(self.protect_writes)
        app.get("/")(self.index)
        app.get("/assets/eva-logo.svg")(self.logo)
        app.get("/assets/eva-task-cjk.ttf")(self.cjk_font)
        app.get("/vendor/<path:name>")(self.vendor)
        app.get("/api/state")(self.state)
        app.post("/api/batches/<batch>/<any(scenes,tasks):kind>")(self.create_plan_record)
        app.put("/api/batches/<batch>/<any(scenes,tasks):kind>/<record_id>")(
            self.update_plan_record
        )
        app.delete("/api/batches/<batch>/<any(scenes,tasks):kind>/<record_id>")(
            self.delete_plan_record
        )
        app.put("/api/batches/<batch>/info")(self.update_info)
        app.get("/api/batches/<batch>/validate")(self.validate)
        app.post("/api/batches/<batch>/import")(self.import_plan)
        app.get("/api/batches/<batch>/export")(self.export_plan)
        app.post("/api/hf/task_sets/publish")(self.publish_hf_task_sets)
        app.post("/api/batches/<batch>/hf/sync")(self.sync_hf_task_set)
        app.post("/api/hf/assets/publish")(self.publish_hf_assets)
        app.post("/api/batches/<batch>/hf/download")(self.download_hf_dataset)
        app.post("/api/batches/<batch>/hf/qc/upload")(self.publish_hf_qc)
        app.post("/api/batches/<batch>/hf/qc/download")(self.download_hf_qc)
        app.get("/api/transfers")(self.transfers_state)
        app.get("/api/transfers/job")(self.transfers_job)
        app.post("/api/transfers")(self.start_transfer)
        app.get("/api/batches/<batch>/qc/export")(self.export_qc)
        app.post("/api/objects")(self.create_object)
        app.put("/api/objects/<object_id>")(self.update_object)
        app.delete("/api/objects/<object_id>")(self.delete_object)
        app.post("/api/objects/<object_id>/photos")(self.upload_photos)
        app.post("/api/objects/import")(self.import_objects)
        app.get("/api/objects/<object_id>/photos/<path:filename>")(self.object_photo)
        app.get("/api/objects/<object_id>/previews/<any(thumb,display):variant>/<path:filename>")(
            self.object_preview
        )
        app.get("/api/review")(self.review)
        app.get("/api/compare")(self.compare)
        app.put("/api/batches/<batch>/episodes/<int:episode_index>/qc")(self.mark_qc)
        app.route(
            "/api/batches/<batch>/episodes/<int:episode_index>/trim",
            methods=["POST", "PUT"],
        )(self.trim_episode)
        app.get("/api/batches/<batch>/episodes/<int:episode_index>/video/<path:key>")(
            self.episode_video
        )
        app.get("/api/robots/<robot_type>/meshes")(self.robot_meshes)
        app.get("/api/robots/<robot_type>/meshes/<path:name>")(self.robot_mesh)
        app.get("/api/batches/<batch>/episodes/<int:episode_index>/transforms")(self.transforms)
        app.errorhandler(ConflictError)(self.conflict)
        app.errorhandler(ValueError)(self.bad_request)
        app.errorhandler(RecordNotFoundError)(self.not_found)
        app.errorhandler(FileNotFoundError)(self.not_found)
        app.errorhandler(HTTPException)(self.http_error)

    def compress_json(self, response: Any) -> Any:
        if (
            response.status_code < 200
            or response.status_code >= 300
            or response.mimetype != "application/json"
            or response.headers.get("Content-Encoding")
            or request.accept_encodings["gzip"] <= 0
        ):
            return response
        content = response.get_data()
        if len(content) < 1024:
            return response
        compressed = gzip.compress(content, compresslevel=4)
        if len(compressed) >= len(content):
            return response
        response.set_data(compressed)
        response.headers["Content-Encoding"] = "gzip"
        response.vary.add("Accept-Encoding")
        return response

    @staticmethod
    def disable_static_cache(response: Any) -> Any:
        """Ensure a restarted editor immediately serves changed UI modules."""
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def protect_writes(self) -> None:
        if request.method not in {"POST", "PUT", "DELETE"}:
            return
        if self.app.config["DATASET_READ_ONLY"]:
            raise Forbidden("Dataset service is read-only")
        if request.headers.get("X-EVA-Dataset-Editor") != "1":
            raise Forbidden("Missing dataset editor request header")
        if request.headers.get("X-EVA-Edit-Mode") != "1":
            raise Forbidden("Enable edit mode before changing dataset files")

    def index(self) -> str:
        return render_template("index.html", locale=self.app.config["DATASET_LOCALE"])

    def logo(self) -> Any:
        return send_file(PROJECT_ROOT / "assets/eva-logo.svg")

    def cjk_font(self) -> Any:
        return send_file(CONSOLE_STATIC_ROOT / "fonts/eva-task-cjk.ttf")

    def vendor(self, name: str) -> Any:
        return send_from_directory(CONSOLE_STATIC_ROOT / "vendor", name)

    def state(self) -> Any:
        return jsonify(
            self.catalog.state(
                request.args.get("batch", ""),
                request.args.get("robot_type", ""),
            )
        )

    def create_plan_record(self, batch: str, kind: str) -> Any:
        return jsonify(self.catalog.upsert_plan(batch, kind, None, request.get_json()))

    def update_plan_record(self, batch: str, kind: str, record_id: str) -> Any:
        return jsonify(self.catalog.upsert_plan(batch, kind, record_id, request.get_json()))

    def delete_plan_record(self, batch: str, kind: str, record_id: str) -> Any:
        return jsonify(self.catalog.delete_plan(batch, kind, record_id))

    def update_info(self, batch: str) -> Any:
        return jsonify(self.catalog.update_info(batch, request.get_json()))

    def validate(self, batch: str) -> Any:
        state = self.catalog.state(batch)
        return jsonify({"issues": state["issues"]})

    def import_plan(self, batch: str) -> Any:
        uploaded = request.files.get("file")
        if uploaded is None:
            raise ValueError("Choose a ZIP file")
        return jsonify(self.catalog.import_plan(batch, uploaded.read()))

    def export_plan(self, batch: str) -> Any:
        stream = self.catalog.export_plan(batch)
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", batch)
        return send_file(
            stream,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{name}.zip",
        )

    def publish_hf_task_sets(self) -> Any:
        """Publish every task set: task sets always sync as a whole."""
        task_sets = {
            path.name: path
            for path in sorted(self.catalog.plans_root.iterdir())
            if path.is_dir() and (path / "tasks.csv").is_file()
        }
        return jsonify({"ok": True, **publish_task_sets(PROJECT_ROOT, task_sets)})

    def publish_hf_assets(self) -> Any:
        return jsonify(publish_assets(PROJECT_ROOT, self.catalog.assets.path))

    def sync_hf_task_set(self, batch: str) -> Any:
        return jsonify(fetch_task_set(PROJECT_ROOT, batch, self.catalog.plans_root))

    def download_hf_dataset(self, batch: str) -> Any:
        info = yaml.safe_load((self.catalog._batch_root(batch) / "info.yaml").read_text())
        destination = self.catalog._source_dataset_dir(batch, info)
        return jsonify(fetch_dataset(PROJECT_ROOT, destination.name, destination.parent))

    def publish_hf_qc(self, batch: str) -> Any:
        info = yaml.safe_load((self.catalog._batch_root(batch) / "info.yaml").read_text())
        dataset_dir = self.catalog._source_dataset_dir(batch, info)
        relative = dataset_dir.relative_to(self.catalog.collection_root).as_posix()
        remote_path = relative if relative.startswith("datasets/") else None
        return jsonify(
            publish_qc(PROJECT_ROOT, dataset_dir, dataset_dir.name, expected_path=remote_path)
        )

    def download_hf_qc(self, batch: str) -> Any:
        info = yaml.safe_load((self.catalog._batch_root(batch) / "info.yaml").read_text())
        dataset_dir = self.catalog._source_dataset_dir(batch, info)
        relative = dataset_dir.relative_to(self.catalog.collection_root).as_posix()
        remote_path = relative if relative.startswith("datasets/") else None
        return jsonify(
            fetch_qc(PROJECT_ROOT, dataset_dir.name, dataset_dir, expected_path=remote_path)
        )

    def transfers_state(self) -> Any:
        return jsonify({"ok": True, "datasets": self.transfers.rows(), **self.transfers.snapshot()})

    def transfers_job(self) -> Any:
        return jsonify({"ok": True, **self.transfers.snapshot()})

    def start_transfer(self) -> Any:
        body = request.get_json(silent=True) or {}
        if body.get("action") == "stop":
            self.transfers.stop(str(body.get("job_id", "")))
            return jsonify({"ok": True})
        names = body.get("datasets")
        if not isinstance(names, list) or not names:
            raise ValueError("Select datasets to transfer")
        self.transfers.start(str(body.get("action", "")), list(dict.fromkeys(names)))
        return jsonify({"ok": True, **self.transfers.snapshot()})

    def export_qc(self, batch: str) -> Any:
        rows = self.catalog.qc_rows(batch)
        stream = io.StringIO(newline="")
        fields = [
            "batch_id",
            "robot_type",
            "task_id",
            "scene_id",
            "slot_id",
            "round_index",
            "round_total",
            "status",
            "episode_index",
            "prompt_en",
            "prompt_zh",
            "qc_reason",
            "qc_note",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        csv_content = stream.getvalue().encode("utf-8-sig")
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", batch)
        content = io.BytesIO()
        supplement_root = f"{name}-supplement"
        with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(f"{name}-qc-missing.csv", csv_content)
            plan_stream = self.catalog.export_qc_plan(batch)
            with zipfile.ZipFile(plan_stream) as plan_archive:
                for member in plan_archive.infolist():
                    filename = Path(member.filename).name
                    if filename in {"info.yaml", "layout.yaml", "scene.csv", "tasks.csv"}:
                        bundle.writestr(f"{supplement_root}/{filename}", plan_archive.read(member))
        content.seek(0)
        return send_file(
            content,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{name}-qc-bundle.zip",
        )

    def create_object(self) -> Any:
        self.catalog.upsert_asset(None, request.get_json())
        return jsonify(self.catalog.state(request.args.get("batch", "")))

    def update_object(self, object_id: str) -> Any:
        self.catalog.upsert_asset(object_id, request.get_json())
        return jsonify(self.catalog.state(request.args.get("batch", "")))

    def delete_object(self, object_id: str) -> Any:
        self.catalog.delete_asset(object_id)
        return jsonify(self.catalog.state(request.args.get("batch", "")))

    def upload_photos(self, object_id: str) -> Any:
        uploads = [
            (uploaded.filename or "photo", uploaded.read())
            for uploaded in request.files.getlist("photos")
        ]
        return jsonify(self.catalog.upload_photos(object_id, uploads))

    def import_objects(self) -> Any:
        uploaded = request.files.get("file") or request.files.get("csv")
        if uploaded is None:
            raise ValueError("Choose an object CSV file")
        photos = [
            (photo.filename or "photo", photo.read()) for photo in request.files.getlist("photos")
        ]
        return jsonify(self.catalog.import_assets_csv(uploaded.read(), photos))

    def object_photo(self, object_id: str, filename: str) -> Any:
        return send_file(
            self.catalog.assets.photo_path(object_id, filename),
            conditional=True,
            max_age=86400,
        )

    def object_preview(self, object_id: str, variant: str, filename: str) -> Any:
        return send_file(
            self.catalog.assets.preview_path(object_id, filename, variant),
            mimetype="image/webp",
            conditional=True,
            max_age=86400,
        )

    def review(self) -> Any:
        return jsonify(
            self.catalog.review(request.args.get("batch", ""), request.args.get("slot_id", ""))
        )

    def compare(self) -> Any:
        return jsonify(
            self.catalog.compare(
                request.args.get("task_id", ""),
                request.args.get("scene_id", ""),
                request.args.get("round_index", 0),
                request.args.get("current_batch", ""),
            )
        )

    def mark_qc(self, batch: str, episode_index: int) -> Any:
        payload = request.get_json(silent=True) or {}
        return jsonify(
            self.catalog.mark_qc(
                batch,
                episode_index,
                str(payload.get("verdict", "")),
                str(payload.get("note", "")),
                str(payload.get("reason", "")),
            )
        )

    def trim_episode(self, batch: str, episode_index: int) -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            start_frame = int(payload["start_frame"])
            end_frame = int(payload["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("start_frame and end_frame must be integers") from error
        return jsonify(
            self.catalog.trim_episode(
                batch,
                episode_index,
                start_frame,
                end_frame,
            )
        )

    def episode_video(self, batch: str, episode_index: int, key: str) -> Any:
        return send_file(
            self.catalog.video_path(batch, episode_index, key),
            conditional=True,
            max_age=30,
        )

    def robot_meshes(self, robot_type: str) -> Any:
        return jsonify(self.catalog.robot_meta(robot_type))

    def robot_mesh(self, robot_type: str, name: str) -> Any:
        return (
            self.catalog.mesh_bytes(robot_type, name),
            200,
            {
                "Content-Type": "application/octet-stream",
                "Cache-Control": "public, max-age=86400",
            },
        )

    def transforms(self, batch: str, episode_index: int) -> Any:
        content, start, total = self.catalog.transform_blob(
            batch,
            episode_index,
            request.args.get("start", 0, type=int),
            request.args.get("count", 120, type=int),
        )
        return (
            content,
            200,
            {
                "Content-Type": "application/octet-stream",
                "X-EVA-Transform-Start": str(start),
                "X-EVA-Transform-Total": str(total),
            },
        )

    def conflict(self, error: ConflictError) -> Any:
        return jsonify({"error": str(error)}), 409

    def bad_request(self, error: ValueError) -> Any:
        return jsonify({"error": str(error)}), 400

    def not_found(self, error: Exception) -> Any:
        label = error.args[0] if error.args else "record"
        return jsonify({"error": f"Not found: {label}"}), 404

    def http_error(self, error: HTTPException) -> Any:
        return jsonify({"error": error.description}), error.code


def create_app(
    plans_root: str | Path = DEFAULT_PLANS_ROOT,
    assets_root: str | Path = DEFAULT_ASSETS_ROOT,
    collection_root: str | Path = DEFAULT_COLLECTION_ROOT,
    read_only: bool = False,
    locale: str = "en",
) -> Flask:
    """Build the dataset management application."""
    if locale not in {"zh", "en"}:
        raise ValueError("locale must be 'zh' or 'en'")
    return DatasetService(plans_root, assets_root, collection_root, read_only, locale).app


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage EVA collection plans and QC")
    parser.add_argument("--plans-root", default=str(DEFAULT_PLANS_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--collection-root", default=str(DEFAULT_COLLECTION_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8416)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument(
        "--language",
        "--locale",
        dest="locale",
        choices=("zh", "en"),
        default="en",
        help="Interface language (default: en)",
    )
    args = parser.parse_args()
    create_app(
        args.plans_root,
        args.assets_root,
        args.collection_root,
        read_only=args.read_only,
        locale=args.locale,
    ).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
