const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(path.join(
  __dirname, "../../src/core/app/console/static/js/scene_map.js",
), "utf8").replace("export function renderSceneGrid", "function renderSceneGrid");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.className = "";
    this.hidden = false;
    this.style = {setProperty() {}};
    this.classes = new Set();
    this.classList = {toggle: (name, enabled) => (
      enabled ? this.classes.add(name) : this.classes.delete(name)
    )};
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [];
    this.append(...children);
  }

  querySelector(selector) {
    return this.children.find((child) => selector.startsWith(".")
      ? child.className.split(" ").includes(selector.slice(1)) : child.tagName === selector);
  }

  querySelectorAll() {
    return this.children;
  }

  getAttribute(name) {
    return name === "src" ? this.src : this.attributes[name];
  }

  removeAttribute(name) {
    delete this[name];
    delete this.attributes[name];
  }
}

function loadRenderer() {
  const context = vm.createContext({
    document: {createElement: (tagName) => new Element(tagName)},
  });
  vm.runInContext(`${SOURCE}\nthis.renderSceneGrid = renderSceneGrid;`, context);
  return context.renderSceneGrid;
}

function makePlan() {
  return {positions: [{position_id: "P1", x: 0, y: 0}], reference_images: [
    {name: "front.jpg", url: "/front.jpg"},
    {name: "side.jpg", url: "/side.jpg"},
  ]};
}

function imageIn(host) {
  return host.children[0].querySelector("a").querySelector("img");
}

function gridScene() {
  return {placements: [{position_id: "P1", object_id: "cup", name_en: "Cup", photo_url: "/cup.jpg"}]};
}

test("每个数据点只显示自身参考图并替代排布网格，切换数据点更新图片", () => {
  const host = new Element();
  const renderSceneGrid = loadRenderer();
  const plan = makePlan();
  renderSceneGrid(host, plan, {reference_image_names: ["side.jpg"]}, null);
  assert.equal(host.children.length, 1);
  assert.equal(host.dataset.displayMode, "reference");
  assert.equal(imageIn(host).src, "/side.jpg");
  assert.equal(host.children.some((child) => child.className.includes("collect-scene-cell")), false);

  renderSceneGrid(host, plan, {reference_image_names: ["front.jpg"]}, null);
  assert.equal(imageIn(host).src, "/front.jpg");
  assert.equal(host.children.length, 1);
});

test("无参考图的数据点显示排布，有图和无图之间切换可恢复", () => {
  const host = new Element();
  const renderSceneGrid = loadRenderer();
  const plan = makePlan();
  renderSceneGrid(host, plan, gridScene(), null);
  assert.match(host.children[0].className, /collect-scene-cell/);
  assert.equal(host.dataset.displayMode, "grid");
  assert.equal(host.children[0].querySelector("img").src, "/cup.jpg");
  renderSceneGrid(host, plan, {reference_image_names: ["front.jpg"]}, null);
  assert.equal(imageIn(host).src, "/front.jpg");
  renderSceneGrid(host, plan, gridScene(), null);
  assert.match(host.children[0].className, /collect-scene-cell/);
  assert.equal(host.dataset.displayMode, "grid");
});

test("参考图加载失败显示提示，仍不显示排布", () => {
  const host = new Element();
  loadRenderer()(host, makePlan(), {reference_image_names: ["front.jpg"]}, null);
  const panel = host.children[0];
  const link = panel.querySelector("a");
  const status = panel.querySelector(".collect-scene-reference-status");
  assert.equal(link.target, "_blank");
  imageIn(host).onerror();
  assert.equal(link.hidden, true);
  assert.equal(status.hidden, false);
  assert.match(status.textContent, /unavailable/);
  assert.equal(host.dataset.displayMode, "reference");
});

test("数据点未绑定参考图时不会显示其他数据点的图库", () => {
  const host = new Element();
  loadRenderer()(host, makePlan(), {...gridScene(), reference_image_names: ["missing.jpg"]}, null);
  assert.match(host.children[0].className, /collect-scene-cell/);
  assert.equal(host.dataset.displayMode, "grid");
});

test("仅 prompt 的数据点不显示网格或其他点参考图，三种模式切换清除旧内容", () => {
  const host = new Element();
  const renderSceneGrid = loadRenderer();
  const plan = makePlan();
  renderSceneGrid(host, plan, {reference_image_names: ["front.jpg"]}, null);
  assert.equal(host.dataset.displayMode, "reference");
  renderSceneGrid(host, plan, {placements: []}, null);
  assert.equal(host.dataset.displayMode, "prompt");
  assert.equal(host.children.length, 0);
  renderSceneGrid(host, plan, gridScene(), null);
  assert.equal(host.dataset.displayMode, "grid");
  assert.equal(host.children[0].querySelector("img").src, "/cup.jpg");
  renderSceneGrid(host, plan, null, null);
  assert.equal(host.dataset.displayMode, "prompt");
  assert.equal(host.children.length, 0);
  renderSceneGrid(host, plan, {reference_image_names: ["front.jpg"]}, null);
  assert.equal(host.dataset.displayMode, "reference");
  assert.equal(imageIn(host).src, "/front.jpg");
});

test("网格物体缺少照片时保留名称，并清除之前照片", () => {
  const host = new Element();
  const renderSceneGrid = loadRenderer();
  renderSceneGrid(host, makePlan(), gridScene(), null);
  const scene = gridScene();
  scene.placements[0].photo_url = "";
  renderSceneGrid(host, makePlan(), scene, null);
  const cell = host.children[0];
  assert.equal(cell.querySelector("img").hidden, true);
  assert.equal(cell.querySelector("img").src, undefined);
  assert.equal(cell.querySelector("b").textContent, "Cup");
});

test("同名参考图更新后按计划版本重新加载图片", () => {
  const host = new Element();
  const renderSceneGrid = loadRenderer();
  const plan = {positions: [], reference_images: [
    {name: "photo", url: "/photo.png"},
  ]};
  const scene = {reference_image_names: ["photo"]};
  renderSceneGrid(host, {...plan, version: "before"}, scene, null);
  const before = imageIn(host);
  renderSceneGrid(host, {...plan, version: "after"}, scene, null);
  assert.notEqual(imageIn(host), before);
});
