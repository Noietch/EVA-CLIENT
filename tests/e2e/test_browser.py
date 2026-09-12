"""Browser smoke tests using local production servers and bundled robot assets."""

import io
import threading
from pathlib import Path

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


def test_collection_slot_click_selects_and_double_click_previews(browser, tmp_path):
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    renderer = source[
        source.index("function renderCollectTiles(") : source.index("function pipeBadge(")
    ]
    page = browser.new_page()
    try:
        page.set_content('<div id="collect-queue-tiles"></div>')
        css = Path(__file__).resolve().parents[2] / "src/core/app/console/static/css/console.css"
        page.add_style_tag(path=str(css))
        page.add_style_tag(content=":root { --danger: #ff0000; --ok: #00ff00; --accent: #ff8800; }")
        page.evaluate("""() => {
            window.S = {collectionSlots: {dataset: 'cup_set', slots: [], active: null},
                STATUS: {collect: {collecting: false}}, collectTaskSelectionPending: false};
            window.$ = (id) => document.getElementById(id);
            window.collectionSlotClickTimer = null;
            window.selected = [];
            window.previews = [];
            window.activateCollectionSlot = (slot) => selected.push(slot.slot_id);
            window.selectCollectionQcTarget = () => {};
            window.selectCollectEpisode = (episode) => previews.push(episode.episode_index);
            window.savedEpisodeId = (episode) => episode?.status === 'saved'
                ? episode.episode_index : null;
        }""")
        page.add_script_tag(content=renderer)
        page.evaluate("""() => {
            S.collectionSlots.slots = [
                {slot_id:'done', dataset:'cup_set', ordinal:0, state:'complete',
                 episode:{status:'saved', episode_index:4, quality:'green'}},
                {slot_id:'red', dataset:'cup_set', ordinal:1, state:'complete',
                 episode:{status:'saved', episode_index:5, quality:'red'}},
                {slot_id:'busy', dataset:'cup_set', ordinal:2, state:'saving'}
            ];
            renderCollectTiles(S.collectionSlots.slots);
        }""")
        tiles = page.locator(".collect-tile")
        assert "SLOT 2" in tiles.nth(1).get_attribute("title")
        assert "EPISODE 5 · RED" in tiles.nth(1).get_attribute("title")
        assert (
            tiles.nth(1).evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(255, 0, 0)"
        )
        assert (
            tiles.nth(0).evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(0, 255, 0)"
        )
        tiles.nth(0).click()
        page.wait_for_function("selected.length === 1")
        assert page.evaluate("selected") == ["done"]
        assert page.evaluate("previews") == []
        tiles.nth(1).dblclick()
        page.wait_for_timeout(350)
        assert page.evaluate("selected") == ["done"]
        assert page.evaluate("previews") == [5]
        assert tiles.nth(2).is_disabled()
        assert "slot-saving" in tiles.nth(2).get_attribute("class")
        assert (
            tiles.nth(2).evaluate("el => getComputedStyle(el).backgroundColor")
            == "rgb(255, 136, 0)"
        )
        page.evaluate(
            "S.collectionSlots.active = S.collectionSlots.slots[1]; "
            "renderCollectTiles(S.collectionSlots.slots)"
        )
        assert "slot-current" in tiles.nth(1).get_attribute("class")
        assert (
            tiles.nth(1).evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(255, 0, 0)"
        )
        page.evaluate(
            "S.STATUS.collect.collecting = true; renderCollectTiles(S.collectionSlots.slots)"
        )
        assert all(tiles.nth(index).is_disabled() for index in range(3))
    finally:
        page.close()


def test_collection_scene_grid_highlights_task_objects(browser):
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    renderer = source[
        source.index("function scenePlanGridPositions(") : source.index("function itemsForPrompt(")
    ]
    page = browser.new_page()
    try:
        page.set_content('<div id="collect-current-scene-grid"></div>')
        css = Path(__file__).resolve().parents[2] / "src/core/app/console/static/css/console.css"
        page.add_style_tag(path=str(css))
        page.evaluate("""() => {
            window.$ = (id) => document.getElementById(id);
            window.S = {SCENE_PLAN: {
                bounds: {width: 500, height: 500},
                positions: [
                    {position_id:'P1', x:0, y:0},
                    {position_id:'P2', x:250, y:0},
                    {position_id:'P3', x:500, y:0},
                    {position_id:'P4', x:0, y:250}
                ],
                tasks: [{
                    task_id:'TASK-130', operation_object:'塑料刀-绿',
                    operation_object_ids: ['knife', 'plate'],
                    prompt_en:'pick up the green knife and place it on the green plate',
                    prompt_zh:'拿起绿色塑料刀并放到绿色小盘子上'
                }]
            }};
            window.scenePlanTask = () => S.SCENE_PLAN.tasks[0];
        }""")
        page.add_script_tag(content=renderer)
        page.evaluate("""() => renderCurrentSceneGrid({placements: [
            {position_id:'P1', object_id:'knife', name:'塑料刀-绿',
             name_zh:'塑料刀-绿', name_en:'Green Knife'},
            {position_id:'P2', object_id:'plate', name:'小盘子-绿',
             name_zh:'小盘子-绿', name_en:'Green Plate'},
            {position_id:'P3', object_id:'utility-knife', name:'美工刀-绿',
             name_zh:'美工刀-绿', name_en:'Green Utility Knife'},
            {position_id:'P4', object_id:'bottle', name:'蓝色水瓶',
             name_zh:'蓝色水瓶', name_en:'Blue Water Bottle'}
        ]})""")

        cells = page.locator(".collect-scene-cell")
        assert cells.nth(0).evaluate("el => el.classList.contains('task-relevant')")
        assert cells.nth(1).evaluate("el => el.classList.contains('task-relevant')")
        assert not cells.nth(2).evaluate("el => el.classList.contains('task-relevant')")
        assert not cells.nth(3).evaluate("el => el.classList.contains('task-relevant')")
        backgrounds = cells.evaluate_all(
            "nodes => nodes.map((node) => getComputedStyle(node).backgroundColor)"
        )
        assert backgrounds[:2] == ["rgb(255, 255, 255)", "rgb(255, 255, 255)"]
        assert backgrounds[2] != "rgb(255, 255, 255)"
        assert backgrounds[3] != "rgb(255, 255, 255)"
    finally:
        page.close()


def test_collection_qc_uses_clicked_episode_after_newer_save(browser):
    source = (
        Path(__file__).resolve().parents[2] / "src/core/app/console/static/js/collect.js"
    ).read_text()
    fragments = [
        source[
            source.index("function selectCollectionQcTarget(") : source.index(
                "function renderCollectionSlotFilters("
            )
        ],
        source[source.index("function renderCollectTiles(") : source.index("function pipeBadge(")],
        source[
            source.index("async function submitEpisodeQc(") : source.index(
                "async function submitQc("
            )
        ],
    ]
    page = browser.new_page()
    try:
        page.set_content("""<div id="collect-queue-tiles"></div><div id="collect-qc-target"></div>
            <div id="collect-qc-status"></div><textarea id="collect-qc-note"></textarea>
            <button id="fail">FAIL</button><button id="note">SAVE NOTE</button>""")
        page.evaluate("""() => {
            window.S = {collectionSlots:{dataset:'set', datasetDir:'/dataset', slots:[]},
                STATUS:{collect:{}}, collectReplayEpisode:99, collectTaskSelectionPending:false};
            window.$ = id => document.getElementById(id);
            window.collectionQcTarget = null;
            window.collectionSlotClickTimer = null;
            window.savedEpisodeId = item => item?.status === 'saved' ? item.episode_index : null;
            window.collectTaskValue = () => 'task';
            window.reviewDatasetFor = () => '/dataset';
            window.historyFor = () => ({episodes:[], queue:[]});
            window.requests = [];
            window.apiPost = async (url, body) => {requests.push(body); return {ok:true};};
            window.clientTrace = window.invalidateEpisodeHistory = window.applyStatus = () => {};
            window.renderCollect = () => {};
            window.apiGet = async () => ({});
            window.pollEpisodeHistory = window.pollCollectionSlots = async () => {};
            window.reviewNoteFor = () => $('collect-qc-note');
            window.episodeQcEndpoint = () => '/api/collect_qc_mark';
            window.activateCollectionSlot = slot => {
                // A poll/selection response now carries a newer episode for this same slot.
                S.collectionSlots.slots = [{...slot, episode:{...slot.episode, episode_index:100}}];
            };
        }""")
        page.add_script_tag(content="\n".join(fragments))
        page.evaluate("""() => {
            renderCollectTiles([{slot_id:'A', dataset:'set', ordinal:0, state:'complete',
                episode:{episode_index:7, status:'saved', quality:'green', task:'task'}}]);
            $('fail').onclick = () => submitEpisodeQc('collect','fail');
            $('note').onclick = () => submitEpisodeNote('collect');
        }""")
        page.locator(".collect-tile").click()
        page.wait_for_function("collectionQcTarget !== null")
        page.locator("#fail").click()
        page.wait_for_function("requests.length === 1")
        page.locator("#collect-qc-note").fill("selected record only")
        page.locator("#note").click()
        page.wait_for_function("requests.length === 2")
        requests = page.evaluate("requests")
        assert [request["episode"] for request in requests] == ["7", "7"]
        assert [request["dataset"] for request in requests] == ["set", "set"]
        assert requests[0]["verdict"] == "fail"
        assert requests[1]["note"] == "selected record only"
        page.evaluate("S.collectionSlots.dataset = 'other'")
        page.locator("#fail").click()
        assert page.evaluate("requests.length") == 2
    finally:
        page.close()


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
                assert ("x5_d405_day" in camera_ids) == (robot == "arx_x5")
                assert ("x5_d405_night" in camera_ids) == (robot == "arx_x5")
            with page.expect_response(lambda response: "/api/devices?" in response.url):
                page.locator("#device-select-camera").select_option("x5_d405_night")
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
            assert page.get_by_label("client / position_scale", exact=True).input_value() == "1.2"
            assert page.locator("#device-select-camera option").evaluate_all(
                "options => options.map(option => option.value)"
            ) == ["yam_d405_orbbec", "yam_orbbec", "none"]
            page.locator("#device-select-camera").select_option("yam_d405_orbbec")
            page.wait_for_function(
                "document.querySelector("
                "'[aria-label=\"settings / camera_auto_exposure_limit_us\"]'"
                ")?.value === '16000'"
            )
            assert page.locator("#device-profile-load").is_hidden()
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
