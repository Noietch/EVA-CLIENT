"""Browser smoke tests using local production servers and bundled robot assets."""

import io
import threading

import numpy as np
import pytest
from PIL import Image
from werkzeug.serving import make_server

import robots  # noqa: F401
from core.config import load_config
from core.devices import REPOSITORY_ROOT, DeviceWorkspace
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene
from tests.integration.web._harness import console_config, serve_console
from tools.datasets.app import create_app

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def viewer_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("robot-viewer")
    app = create_app(root / "plans", root / "assets", root / "collection", read_only=True)

    @app.get("/robot-preview/<robot_type>")
    def preview(robot_type):
        return (
            '''<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body,main{margin:0;width:100%;height:100%;overflow:hidden}canvas{display:block;width:100%;height:100%}</style>
<script type="importmap">{"imports":{
"three":"/vendor/three/three.module.min.js","three/addons/":"/vendor/three/addons/"
}}</script>
</head><body><main><canvas></canvas><div id="empty"></div></main>
<script type="module">
import {loadThree,RobotViewer} from '/static/robot-viewer.js';
import * as THREE from 'three';
await loadThree();
window.viewer=new RobotViewer(document.querySelector('canvas'),document.querySelector('#empty'));
await viewer.load("'''
            + robot_type
            + """",{batch_id:''},null);
window.THREE=THREE;
window.ready=Object.values(viewer.meshes).reduce((n,arm)=>n+Object.keys(arm).length,0);
</script></body></html>"""
        )

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("viewport", [(1440, 1000), (390, 844)], ids=["desktop", "mobile"])
@pytest.mark.parametrize("robot_type", ["dual_yam", "arx_x5"])
def test_robot_assets_render_and_move(browser, viewer_url, tmp_path, viewport, robot_type):
    width, height = viewport
    page = browser.new_page(viewport={"width": width, "height": height})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(f"{viewer_url}/robot-preview/{robot_type}", wait_until="networkidle")
        page.wait_for_function("window.ready === 18")
        assert page.evaluate("""() => {
          const parent=viewer.canvas.parentElement;
          parent.style.display='none'; viewer.resize();
          const finite=viewer.camera.position.toArray().every(Number.isFinite);
          parent.style.display=''; viewer.resize();
          return finite;
        }""")
        page.wait_for_timeout(200)
        image = np.asarray(Image.open(io.BytesIO(page.screenshot(path=tmp_path / "initial.png"))))
        assert np.sum(image[:, :, :3].mean(axis=2) < 180) > 2000
        assert page.evaluate("""() => {
          viewer.scene.updateMatrixWorld(true);
          for(const arm of Object.values(viewer.meshes)) for(const mesh of Object.values(arm)) {
            const points=mesh.geometry.attributes.position;
            for(let i=0;i<points.count;i++) {
              const p=new THREE.Vector3().fromBufferAttribute(points,i)
                .applyMatrix4(mesh.matrixWorld).project(viewer.camera);
              if(Math.abs(p.x)>=1 || Math.abs(p.y)>=1) return false;
            }
          }
          return true;
        }""")
        robot = ROBOT_REGISTRY.build(robot_type)
        qpos = robot.initial_qpos.copy()
        qpos[0] += 0.35
        qpos[6] = 0
        page.evaluate(
            "transforms=>viewer.applyInitial(transforms)", UrdfScene(robot).transforms(qpos)
        )
        page.wait_for_timeout(200)
        moved = np.asarray(Image.open(io.BytesIO(page.screenshot())))
        assert np.sum(np.any(image != moved, axis=2)) > 500
        page.mouse.move(width / 2, height / 2)
        page.mouse.down()
        page.mouse.move(width * 0.65, height * 0.55, steps=10)
        page.mouse.up()
        page.wait_for_timeout(300)
        rotated = np.asarray(Image.open(io.BytesIO(page.screenshot(path=tmp_path / "rotated.png"))))
        assert np.sum(np.any(moved != rotated, axis=2)) > 500
        assert not errors, errors
    finally:
        page.close()


def test_device_panel_switches_robot_options_and_vr_parameters(browser, tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_WORKSTATION_PATH", str(tmp_path / "workstation.yaml"))
    with serve_console(console_config()) as console:
        page = browser.new_page()
        page.route("**/api/camera/**", lambda route: route.abort())
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(f"http://127.0.0.1:{console.port}", wait_until="domcontentloaded")
            page.wait_for_function(
                "document.querySelector('[data-tab=rl]').classList.contains('disabled')"
            )
            with page.expect_response(lambda response: response.url.endswith("/api/tab_switch")):
                page.locator("button[data-tab=manual]").click()
            console.pump()
            page.locator("#device-select-robot").wait_for()
            for robot, opened in (("agibot_g2", "0"), ("agilex_piper", "0.1"), ("arx_x5", "1")):
                with page.expect_response(lambda response: "/api/devices?" in response.url):
                    page.locator("#device-select-robot").select_option(robot)
                page.wait_for_function(
                    "robot => document.querySelector('#device-select-robot')?.value === robot",
                    arg=robot,
                )
                if page.locator("#device-select-teleop").input_value() != "vr_webxr":
                    with page.expect_response(lambda response: "/api/devices?" in response.url):
                        page.locator("#device-select-teleop").select_option("vr_webxr")
                page.wait_for_function(
                    "value => document.querySelector("
                    "'[aria-label=\"client / gripper / open_value\"]')?.value === value",
                    arg=opened,
                )
                camera_ids = page.locator("#device-select-camera option").evaluate_all(
                    "options => options.map(option => option.value)"
                )
                assert "yam_d405" not in camera_ids
                assert ("x5_d405" in camera_ids) == (robot == "arx_x5")
            with page.expect_response(lambda response: "/api/devices?" in response.url):
                page.locator("#device-select-camera").select_option("x5_d405")
            with page.expect_response(lambda response: "/api/devices?" in response.url):
                page.locator("#device-select-robot").select_option("agibot_g2")
            page.wait_for_function(
                "document.querySelector('#device-select-camera')?.value === 'external'"
            )
            assert not errors, errors
        finally:
            page.close()


def test_console_tabs_change_backend_state(browser, tmp_path, monkeypatch):
    monkeypatch.setenv("EVA_WORKSTATION_PATH", str(tmp_path / "workstation.yaml"))
    with serve_console(console_config()) as console:
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(f"http://127.0.0.1:{console.port}", wait_until="domcontentloaded")
            page.wait_for_function(
                "document.querySelector('[data-tab=rl]').classList.contains('disabled')"
            )
            page.screenshot(path=tmp_path / "console.png")
            assert not errors, errors
            for tab in ("collect", "rl"):
                button = page.locator(f"button[data-tab={tab}]")
                assert "disabled" in button.get_attribute("class").split()
                button.click()
                assert console.runtime.console_ctx.active_tab == "debug"
            for tab in ("manual", "replay", "eval", "debug"):
                with page.expect_response(
                    lambda response: response.url.endswith("/api/tab_switch")
                ) as pending:
                    page.locator(f"button[data-tab={tab}]").click()
                assert pending.value.ok
                page.wait_for_function(
                    "tab=>document.querySelector('button.tab.active')?.dataset.tab===tab", arg=tab
                )
                console.pump()
                assert console.runtime.console_ctx.active_tab == tab
                if tab == "manual":
                    for _ in range(10):
                        console.pump()
                        page.wait_for_timeout(100)
                        if page.locator(".ms-j").count():
                            break
                    names = [
                        name
                        for group in console.runtime.robot.actuator_groups
                        for name in group.joint_names
                    ]
                    assert page.locator(".ms-j").all_text_contents() == names
                    assert page.locator(".ms-group").all_text_contents() == [
                        group.name for group in console.runtime.robot.actuator_groups
                    ]
                    page.set_viewport_size({"width": 390, "height": 844})
                    page.screenshot(path=tmp_path / "device-mobile.png")
                    assert page.locator(".ms-j").evaluate_all(
                        "nodes => nodes.every(node => node.scrollWidth <= node.clientWidth)"
                    )
                    page.set_viewport_size({"width": 1440, "height": 1000})
                    page.screenshot(path=tmp_path / "device-desktop.png")
            assert console.get("/api/status").status == 200
            assert not errors, errors
        finally:
            page.close()

    workspace = DeviceWorkspace(tmp_path / "workstation.yaml")
    selected = dict(robot="dual_yam", teleop="vr_webxr", camera="external")
    workspace.save(selected, workspace.resolve(selected))
    vr = workspace.configure(load_config(REPOSITORY_ROOT / "configs/00_base/defaults.py"))
    with serve_console(console_config(robot_type="dual_yam", collection=vr.collection)) as console:
        console.runtime.console_ctx.scene = UrdfScene(console.runtime.robot)
        page = browser.new_page()
        try:
            page.goto(f"http://127.0.0.1:{console.port}", wait_until="domcontentloaded")
            page.locator("button[data-tab=manual]").click()
            console.pump()
            page.wait_for_selector("#device-select-robot")
            assert page.locator("#control-teleop-title").inner_text() == "VR"
            assert page.locator("#gb-step").inner_text() == "DEVICE"
            assert page.locator("#manual-panel-tune").is_hidden()
            assert page.locator("#device-select-robot").input_value() == "dual_yam"
            assert page.locator("#device-select-teleop").input_value() == "vr_webxr"
            assert page.get_by_label("client / position_scale", exact=True).input_value() == "1"
            page.locator("#device-select-camera").select_option("yam_d405")
            page.wait_for_selector("#device-profile-load", state="visible")
            page.locator("#device-profile-load").click()
            page.wait_for_function(
                "document.querySelector('#device-profile-values').textContent.includes('33000')"
            )
            assert not console.runtime.collection_teleop_armed
            page.set_viewport_size({"width": 390, "height": 844})
            page.locator("#gl").scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            canvas = page.locator("#gl")
            bounds = canvas.bounding_box()
            assert 300 < bounds["width"] <= 390 and bounds["x"] >= 0
            pixels = np.asarray(
                Image.open(io.BytesIO(canvas.screenshot(path=tmp_path / "device-robot-mobile.png")))
            )
            assert np.sum(pixels[:, :, :3].mean(axis=2) < 120) > 2000
        finally:
            page.close()
