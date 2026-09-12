"""Serve the task-plan, object-asset, and collection QC browser.

Plans are discovered as batches below task_sets. Objects live in one shared
asset catalog, while collected episodes remain in their existing LeRobot
directories and are joined to plans through stable slot IDs.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import Forbidden, HTTPException

from tools.datasets.collection import PlanCatalog
from tools.datasets.store import ConflictError, RecordNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets/data_collection"
DEFAULT_PLANS_ROOT = DEFAULT_DATA_ROOT / "task_sets"
DEFAULT_ASSETS_ROOT = DEFAULT_DATA_ROOT / "assets"
DEFAULT_COLLECTION_ROOT = DEFAULT_DATA_ROOT
CONSOLE_STATIC_ROOT = PROJECT_ROOT / "src/core/app/console/static"


class DatasetService:
    """Own the dataset catalog and its HTTP endpoints."""

    def __init__(self, plans_root, assets_root, collection_root, read_only):
        self.app = app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
        app.config["DATASET_READ_ONLY"] = read_only
        self.catalog = PlanCatalog(plans_root, assets_root, collection_root)

        # Register the catalog and review endpoints
        app.after_request(self.compress_json)
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
        app.post("/api/objects")(self.create_object)
        app.put("/api/objects/<object_id>")(self.update_object)
        app.delete("/api/objects/<object_id>")(self.delete_object)
        app.post("/api/objects/<object_id>/photos")(self.upload_photos)
        app.get("/api/objects/<object_id>/photos/<path:filename>")(self.object_photo)
        app.get("/api/objects/<object_id>/previews/<any(thumb,display):variant>/<path:filename>")(
            self.object_preview
        )
        app.get("/api/review")(self.review)
        app.put("/api/batches/<batch>/episodes/<int:episode_index>/qc")(self.mark_qc)
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
        return render_template("index.html")

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
        return jsonify(self.catalog.assets.save_photos(object_id, uploads))

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

    def mark_qc(self, batch: str, episode_index: int) -> Any:
        payload = request.get_json(silent=True) or {}
        return jsonify(
            self.catalog.mark_qc(
                batch,
                episode_index,
                str(payload.get("verdict", "")),
                str(payload.get("note", "")),
            )
        )

    def episode_video(self, batch: str, episode_index: int, key: str) -> Any:
        return send_file(self.catalog.video_path(batch, episode_index, key), conditional=True)

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
) -> Flask:
    """Build the dataset management application."""
    return DatasetService(plans_root, assets_root, collection_root, read_only).app


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage EVA collection plans and QC")
    parser.add_argument("--plans-root", default=str(DEFAULT_PLANS_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--collection-root", default=str(DEFAULT_COLLECTION_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8418)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    create_app(
        args.plans_root,
        args.assets_root,
        args.collection_root,
        read_only=args.read_only,
    ).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
