// A task shows its authored reference photo; a task without one shows only its prompt.
export function renderSceneReference(host, plan, task) {
  const url = task?.reference_image_url || "";
  host.hidden = !url;
  if (!url) {
    host.replaceChildren();
    delete host.dataset.referenceKey;
    return;
  }
  const referenceKey = JSON.stringify([plan.version, url]);
  if (host.dataset.referenceKey === referenceKey && host.children.length) return;
  const panel = document.createElement("div");
  const label = document.createElement("span");
  const link = document.createElement("a");
  const image = document.createElement("img");
  const status = document.createElement("span");
  panel.className = "collect-scene-reference";
  label.className = "collect-scene-reference-label";
  label.textContent = "参考图 / REFERENCE IMAGE";
  link.className = "collect-scene-reference-link";
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  image.className = "collect-scene-reference-image";
  image.alt = "当前任务参考图";
  status.className = "collect-scene-reference-status";
  status.textContent = "图片加载失败 / Image unavailable";
  status.hidden = true;
  image.onerror = () => {
    link.hidden = true;
    status.hidden = false;
  };
  link.append(image);
  panel.append(label, link, status);
  host.replaceChildren(panel);
  host.dataset.referenceKey = referenceKey;
  image.src = url;
}
