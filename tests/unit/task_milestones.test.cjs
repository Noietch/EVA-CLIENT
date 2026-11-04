const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(
  path.join(__dirname, "../../tools/datasets/static/entity-ui.js"),
  "utf8",
);
const START = SOURCE.indexOf("function buildTaskMilestoneEditor()");
const END = SOURCE.indexOf("function buildTaskObjectPicker()", START);
const MILESTONE_EDITOR = SOURCE.slice(START, END);

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.disabled = false;
    this.value = "";
    this.textContent = "";
    this.listeners = {};
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = children;
  }

  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }

  emit(type) {
    for (const listener of this.listeners[type] || []) listener({target: this});
  }
}

function harness(milestones) {
  const state = {draft: {tasks: {milestones}}};
  const context = vm.createContext({
    app: state,
    translate: (key) => key,
    node: (tagName, className = "", text = "") => {
      const element = new Element(tagName);
      element.className = className;
      element.textContent = text;
      return element;
    },
    input: (type, name = "", value = "") => {
      const element = new Element("input");
      element.type = type;
      element.name = name;
      element.value = value;
      return element;
    },
    button: (label) => {
      const element = new Element("button");
      element.textContent = label;
      element.click = () => element.emit("click");
      return element;
    },
    field: (label, control) => {
      const wrapper = new Element("label");
      wrapper.append(new Element("span"), control);
      return wrapper;
    },
  });
  vm.runInContext(`${MILESTONE_EDITOR}\nthis.build = buildTaskMilestoneEditor;`, context);
  return {state, surface: context.build()};
}

function row(surface, index) {
  return surface.children[index + 2];
}

function controls(rowElement) {
  return {
    id: rowElement.children[0].children[1],
    label: rowElement.children[1].children[1],
    up: rowElement.children[2].children[0],
    down: rowElement.children[2].children[1],
    remove: rowElement.children[2].children[2],
  };
}

function snapshot(state) {
  return JSON.parse(JSON.stringify(state.draft.tasks.milestones));
}

test("milestone editor adds, edits, reorders, and removes milestones", () => {
  const milestones = [{id: "grasp", label: "Grasp object"}];
  const {state, surface} = harness(milestones);
  const add = surface.children[0].children[1];

  add.click();
  assert.deepEqual(snapshot(state), [
    {id: "grasp", label: "Grasp object"}, {id: "", label: ""},
  ]);
  const second = controls(row(surface, 1));
  second.id.value = "place";
  second.id.emit("input");
  second.label.value = "Place object";
  second.label.emit("input");
  assert.deepEqual(snapshot(state), [
    {id: "grasp", label: "Grasp object"}, {id: "place", label: "Place object"},
  ]);

  const first = controls(row(surface, 0));
  assert.equal(first.up.disabled, true);
  first.down.click();
  assert.deepEqual(snapshot(state), [
    {id: "place", label: "Place object"}, {id: "grasp", label: "Grasp object"},
  ]);

  controls(row(surface, 1)).up.click();
  assert.deepEqual(snapshot(state), [
    {id: "grasp", label: "Grasp object"}, {id: "place", label: "Place object"},
  ]);
  const restoredLast = controls(row(surface, 1));
  assert.equal(restoredLast.down.disabled, true);
  restoredLast.remove.click();
  assert.deepEqual(snapshot(state), [{id: "grasp", label: "Grasp object"}]);
});

