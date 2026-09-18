// Standard UI glossary for the console. Each entry is [Chinese, English].
const GLOSSARY = {
  "app.title.bar": ["EVA · 客户端", "EVA · CLIENT"],
  "tabs.dashboard": ["仪表盘", "DASHBOARD"],
  "tabs.debug": ["调试", "DEBUG"],
  "tabs.device": ["设备", "DEVICE"],
  "tabs.collect": ["采集", "COLLECT"],
  "tabs.replay": ["回放", "REPLAY"],
  "tabs.rl": ["强化学习", "RL"],
  "tabs.eval": ["评测", "EVAL"],
  "tabs.result": ["结果", "RESULT"],
  "guide.step": ["步骤 {step}/4", "STEP {step}/4"],
  "guide.policy": ["策略 —", "policy —"],
  "guide.noEpisode": ["未选择回合", "no episode selected"],
  "guide.hint": ["按左侧步骤 1 → 4 执行", "Follow steps 1 → 4 on the left to run"],
  "dashboard.header": ["原始数据运维", "RAW DATA OPERATIONS"],
  "dashboard.title": ["运维看板", "Operations dashboard"],
  "dashboard.desc": ["原始片段吞吐、需求与质量趋势", "Raw episode throughput, demand, and quality over time."],
  "dashboard.scope": ["仅原始数据", "RAW ONLY"],
  "dashboard.dateRange": ["看板时间范围", "DATE RANGE"],
  "dashboard.from": ["起始", "FROM"],
  "dashboard.to": ["截止", "TO"],
  "dashboard.apply": ["应用", "APPLY"],
  "dashboard.reset": ["重置时间范围", "Reset date range"],
  "dashboard.overview": ["数据集概览", "Dataset overview"],
  "dashboard.overviewDesc": ["原始输出、效率与任务需求", "Raw data output, efficiency, and task demand"],
  "dashboard.allData": ["全部数据", "ALL DATA"],
  "dashboard.recorded": ["已记录数据", "DATA RECORDED"],
  "dashboard.episodes": ["回合数", "EPISODES"],
  "dashboard.frames": ["帧数", "FRAMES"],
  "dashboard.avgDuration": ["平均时长", "AVG DURATION"],
  "dashboard.efficiency": ["效率", "EFFICIENCY"],
  "dashboard.validRate": ["通过率", "VALID RATE"],
  "dashboard.uploaded": ["已上传", "UPLOADED"],
  "dashboard.notUploaded": ["未上传", "NOT UPLOADED"],
  "dashboard.trend": ["输出趋势", "OUTPUT TREND"],
  "dashboard.byDay": ["按日汇总采集", "Collection by day"],
  "dashboard.health": ["流水线健康", "PIPELINE HEALTH"],
  "dashboard.usable": ["可用输出", "Usable output"],
  "dashboard.delivery": ["数据交付", "DATA DELIVERY"],
  "dashboard.uploadSync": ["上传同步", "Upload sync"],
  "dashboard.notConfigured": ["未配置", "NOT CONFIGURED"],
  "dashboard.local": ["本地", "LOCAL"],
  "dashboard.remote": ["远端", "REMOTE"],
  "dashboard.scanTarget": ["扫描目标", "SCAN TARGET"],
  "dashboard.new": ["新增", "NEW"],
  "dashboard.changed": ["已变更", "CHANGED"],
  "dashboard.same": ["未变", "SAME"],
  "dashboard.remove": ["移除", "REMOVE"],
  "dashboard.noExport": ["未找到可用导出", "No dataset export found"],
  "dashboard.taskDemand": ["任务需求", "TASK DEMAND"],
  "dashboard.requirements": ["采集要求", "Collection requirements"],
  "dashboard.progress": ["进度", "PROGRESS"],
  "dashboard.status": ["状态", "STATUS"],
  "dashboard.noTasks": ["未找到采集任务", "No collection tasks found"],
  "dashboard.noEpisodes": ["未找到原始采集回合", "No raw collection episodes found"],
  "dashboard.evalOverview": ["评测概览", "Evaluation overview"],
  "dashboard.successRate": ["成功率", "SUCCESS RATE"],
  "dashboard.trialLog": ["试次日志", "TRIAL LOG"],
  "dashboard.evalActivity": ["评测活动", "Eval activity"],
  "dashboard.noTrials": ["未找到原始评测回合", "No raw eval trials found"],
  "debug.prompt": ["提示词", "PROMPT"],
  "debug.config": ["配置", "CONFIG"],
  "debug.mode": ["模式", "MODE"],
  "debug.strategy": ["策略", "STRATEGY"],
  "debug.setup": ["设置", "SETUP"],
  "debug.control": ["控制", "CONTROL"],
  "debug.gripper": ["夹爪控制", "GRIPPER"],
  "debug.tuning": ["调参", "TUNING"],
  "debug.selectTask": ["先选择任务", "↑ select a task first"],
  "debug.preparing": ["自动 · 正在准备机器人…", "AUTO · PREPARING ROBOT…"],
  "debug.pause": ["暂停", "PAUSE ‖"],
  "debug.resume": ["继续", "RESUME ▶"],
  "debug.retry": ["重试", "RETRY ↻"],
  "debug.run": ["运行 ▶", "RUN ▶"],
  "debug.reset": ["重置 ⟲", "RESET ⟲"],
  "debug.idle": ["空闲", "IDLE"],
  "debug.sim": ["仿真 ↻", "SIM ↻"],
  "debug.real": ["真实 ▶", "REAL ▶"],
  "debug.stop": ["停止 ■", "STOP ■"],
  "debug.apply": ["应用 ⤓", "APPLY ⤓"],
  "collect.dataset": ["数据集", "DATASET"],
  "collect.datasetName": ["数据集名称", "DATASET NAME"],
  "collect.record": ["录制", "RECORD"],
  "collect.motion": ["运动", "MOTION"],
  "collect.locked": ["已锁定", "LOCKED"],
  "collect.startRecord": ["开始录制", "START RECORD"],
  "collect.cancel": ["取消", "CANCEL"],
  "collect.home": ["回零点", "HOME"],
  "collect.slots": ["采集槽位", "COLLECTION SLOTS"],
  "collect.complete": ["已完成", "COMPLETE"],
  "collect.all": ["全部", "ALL"],
  "collect.scene": ["场景", "SCENE"],
  "collect.task": ["任务", "TASK"],
  "collect.pending": ["待处理", "PENDING"],
  "collect.repair": ["待修复", "REPAIR"],
  "collect.qc": ["质量检测", "QUALITY CHECK"],
  "collect.selected": ["已选择", "SELECTED"],
  "collect.pass": ["通过 ✓", "PASS ✓"],
  "collect.fail": ["不通过 ✗", "FAIL ✗"],
  "collect.saveNote": ["保存说明", "SAVE NOTE"],
  "collect.export": ["质量导出", "QUALITY EXPORT"],
  "collect.exportFormat": ["导出格式", "EXPORT FORMAT"],
  "collect.exportDataset": ["导出数据集", "EXPORT DATASET"],
  "collect.enableMotion": ["启用采集运动", "Enable collection motion"],
  "collect.currentPositions": ["当前场景位置", "Current scene positions"],
  "collect.previousSet": ["上一组数据集", "Previous dataset set"],
  "collect.nextSet": ["下一组数据集", "Next dataset set"],
  "collect.vrReviewAria": ["采集数据：左摇杆移动光标即选中采集位置、双击回放、拨动返回实时，左手 Y 键切换通过或不合格", "Collection data: the left stick moves one cursor that selects the capture slot, double-press replays, flick returns live, and the left-hand Y key switches pass or fail"],
  "collect.vrReviewHint": ["左摇杆：移动即选中采集位置、边缘自动翻页 · 双击下压：回放 · 回放时拨动：返回实时 · Y：PASS ↔ FAIL", "Left stick: move to select the capture slot and turn pages at the edge · double-press: replay · flick during replay: return live · Y: PASS ↔ FAIL"],
  "collect.error.selectPosition": ["选择采集位置失败：{error}", "Failed to select collection position: {error}"],
  "collect.error.selectFirst": ["请先用左摇杆选择采集位置", "Use the left stick to select a collection position first"],
  "collect.error.waitSave": ["请等待当前采集或保存完成后再选择采集位置", "Wait for the current capture or save to finish before selecting a collection position"],
  "collect.qc.selectSaved": ["先选择已保存的数据", "Select saved data first"],
  "collect.error.statusSwitch": ["状态切换失败：{error}", "Failed to switch status: {error}"],
  "collect.error.page": ["翻页失败：{error}", "Failed to change page: {error}"],
  "replay.episode": ["回合", "EPISODE"],
  "replay.joint": ["关节", "joint"],
  "replay.eef": ["末端", "eef"],
  "replay.load": ["加载 ▶", "LOAD ▶"],
  "replay.next": ["下一回合 ▶▶", "NEXT EP ▶▶"],
  "replay.saveAnnotation": ["保存标注", "SAVE ANNOTATION"],
  "replay.datasetDir": ["数据集路径（例如 ...）", "dataset dir (e.g. datasets/sft/.../push_black_block)"],
  "replay.episodeId": ["回合编号", "episode id"],
  "replay.qcNote": ["质量说明（可选）", "QC note (optional)"],
  "replay.annotation": ["语言标注", "language annotation"],
  "replay.configSetup": ["配置与设置", "CONFIG & SETUP"],
  "replay.loadFirst": ["先加载一个回合", "↑ load an episode first"],
  "replay.reset": ["重置 ⟲", "RESET ⟲"],
  "replay.tuning": ["调参", "TUNING"],
  "rl.task": ["任务", "TASK"],
  "rl.models": ["模型", "MODELS"],
  "rl.notSelected": ["未选择", "NOT SELECTED"],
  "rl.optional": ["可选", "OPTIONAL"],
  "rl.dataConfig": ["数据配置", "DATA CONFIG"],
  "rl.format": ["格式", "FORMAT"],
  "rl.dataset": ["数据集", "dataset"],
  "rl.reward": ["奖励", "REWARD"],
  "rl.retrySetup": ["重试设置", "RETRY SETUP ↻"],
  "rl.rolloutHil": ["演练 + 人在回路", "ROLLOUT + HIL"],
  "rl.source": ["来源", "SOURCE"],
  "rl.teleop": ["遥控", "teleop"],
  "rl.stopIntervene": ["停止 / 干预", "STOP / INTERVENE"],
  "rl.accept": ["接受", "ACCEPT"],
  "rl.abandon": ["放弃", "ABANDON"],
  "rl.savedData": ["已保存数据", "SAVED DATA"],
  "rl.saveRollout": ["保存回放", "SAVE ROLLOUT"],
  "rl.replay": ["回放 ▶", "REPLAY ▶"],
  "rl.expand": ["展开", "EXPAND"],
  "eval.task": ["任务", "TASK"],
  "eval.model": ["模型", "MODEL"],
  "eval.trial": ["回合", "TRIAL"],
  "eval.setup": ["设置", "SETUP ⚙"],
  "eval.run": ["运行 ▶", "RUN ▶"],
  "eval.score": ["评分", "SCORE"],
  "eval.saveNext": ["保存分数 · 下一项", "SAVE SCORE · NEXT"],
  "eval.awaiting": ["等待试次", "AWAITING TRIAL…"],
  "eval.setupFailed": ["设置失败", "SETUP FAILED"],
  "eval.robotReady": ["机器人就绪", "ROBOT READY"],
  "eval.selectTask": ["选择任务", "SELECT TASK"],
  "eval.selectTrial": ["先选择回合", "↑ select a trial first"],
  "eval.noModels": ["尚无记录模型", "no recorded models yet"],
  "eval.noNote": ["（无注释）", "(no note)"],
  "eval.noCamera": ["无摄像头视频", "no camera video"],
  "eval.scored": ["得分：{score}", "scored: {score}"],
  "manual.devices": ["设备", "DEVICES"],
  "manual.killAll": ["停止全部", "KILL ALL"],
  "manual.dispatch": ["调度", "DISPATCH"],
  "manual.connect": ["连接", "CONNECT"],
  "manual.sendReal": ["发送到真实环境 ▶", "SEND TO REAL ▶"],
  "manual.home": ["回零点 ⌂", "HOME ⌂"],
  "stage.loading3d": ["加载中三维模型…", "LOADING 3D…"],
  "stage.fetching": ["正在读取机器人模型", "fetching URDF meshes"],
  "stage.live": ["实时", "LIVE"],
  "stage.review": ["复核", "REVIEW"],
  "stage.returnLive": ["返回实时", "RETURN TO LIVE"],
  "stage.replayRobot": ["在机器人上回放", "REPLAY ON ROBOT"],
  "chart.critic": ["评价值", "CRITIC VALUE"],
  "chart.timeValue": ["时间（秒）· 数值", "time (s) · value"],
  "chart.action": ["动作", "ACTION"],
  "chart.state": ["状态", "STATE"],
  "chart.all": ["全部", "ALL"],
  "chart.expand": ["展开", "expand"],
  "chart.close": ["关闭", "CLOSE ✕"],
  "error.robotReplay": ["无法打开机器人回放", "Unable to open robot replay"],
  "state.enabled": ["已启用", "ENABLED"],
  "state.locked": ["已锁定", "LOCKED"],
  "state.running": ["运行中", "running"],
  "state.starting": ["启动中…", "STARTING…"],
  "state.stopping": ["停止中…", "STOPPING…"],
  "state.unavailable": ["不可用", "UNAVAILABLE"],
  "state.converting": ["转换中", "CONVERTING"],
  "state.saved": ["已保存 ✓", "SAVED ✓"],
  "state.noSaved": ["无已保存片段", "no saved episodes"],
  "state.noAnnotation": ["暂无注释", "no annotation yet"],
  "action.scanning": ["扫描中…", "SCANNING..."],
  "action.syncing": ["同步中…", "SYNCING..."],
  "action.confirmSync": ["确认同步", "CONFIRM SYNC"],
  "action.applied": ["已应用", "applied"],
  "action.applying": ["应用中…", "applying…"],
  "action.failed": ["失败", "failed"],
  "word.episode": ["回合", "episode"],
  "word.episodes": ["回合", "episodes"],
  "word.trial": ["试次", "trial"],
  "word.trials": ["试次", "trials"],
  "word.task": ["任务", "task"],
  "word.tasks": ["任务", "tasks"],
  "word.loading": ["加载中", "loading"],
  "word.ready": ["就绪", "ready"],
  "word.running": ["运行中", "running"],
  "word.pending": ["待处理", "pending"],
  "word.failed": ["失败", "failed"],
  "word.select": ["选择", "select"],
  "word.save": ["保存", "save"],
  "word.saved": ["已保存", "saved"],
  "word.note": ["说明", "note"],
  "word.valid": ["有效", "valid"],
  "word.frames": ["帧", "frames"],
  "word.uploaded": ["已上传", "uploaded"],
  "word.scanning": ["扫描中", "scanning"],
  "word.syncing": ["同步中", "syncing"],
  "word.connected": ["已连接", "connected"],
  "word.disconnected": ["未连接", "disconnected"],
  "word.online": ["在线", "online"],
  "word.offline": ["离线", "offline"],
  "word.error": ["错误", "error"],
  "word.unknown": ["未知", "unknown"],
  "word.camera": ["摄像头", "camera"],
  "word.video": ["视频", "video"],
};

const locale = "en";
const index = new Map();
const phrases = [];
const templates = [];
for (const [key, values] of Object.entries(GLOSSARY)) {
  index.set(values[0], key);
  index.set(values[1], key);
  if (values[1].includes("{")) templates.push(values);
  else phrases.push(values);
}
phrases.sort((left, right) => right[1].length - left[1].length);

export function t(key, variables = {}) {
  const values = GLOSSARY[key];
  if (!values) return key;
  let text = values[locale === "zh" ? 0 : 1];
  for (const [name, value] of Object.entries(variables)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}

function translateText(value) {
  const trimmed = value.trim();
  const key = index.get(trimmed);
  if (key) {
    const translated = t(key);
    return value === trimmed ? translated : value.replace(trimmed, translated);
  }
  let translated = value;
  for (const values of templates) {
    const source = locale === "zh" ? values[1] : values[0];
    const target = locale === "zh" ? values[0] : values[1];
    const names = [...source.matchAll(/\{([^}]+)\}/g)].map((match) => match[1]);
    const parts = source.split(/\{[^}]+\}/g);
    const pattern = parts.map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("(.+?)");
    translated = translated.replace(new RegExp(pattern, "gi"), (...matches) => {
      let output = target;
      names.forEach((name, index) => { output = output.replace(`{${name}}`, matches[index + 1]); });
      return output;
    });
  }
  for (const values of phrases) {
    const source = locale === "zh" ? values[1] : values[0];
    const target = locale === "zh" ? values[0] : values[1];
    if (!source || source === target || !translated.toLocaleLowerCase().includes(source.toLocaleLowerCase())) continue;
    translated = translated.replace(new RegExp(source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), target);
  }
  return translated;
}

function translateNode(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (node.parentElement?.closest('script, style, [translate="no"]')) continue;
    const translated = translateText(node.nodeValue || "");
    if (translated !== node.nodeValue) node.nodeValue = translated;
  }
}

function translateAttributes(root) {
  const elements = root.querySelectorAll ? root.querySelectorAll("*") : [];
  for (const element of elements) {
    if (element.closest('[translate="no"]')) continue;
    for (const attribute of ["title", "aria-label", "placeholder"]) {
      const value = element.getAttribute(attribute);
      if (!value) continue;
      const translated = translateText(value);
      if (translated !== value) element.setAttribute(attribute, translated);
    }
  }
}

export function applyLocale(root = document) {
  translateNode(root);
  translateAttributes(root);
}

export function initLocale() {
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  applyLocale(document);
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      const parent = record.target.nodeType === Node.ELEMENT_NODE ? record.target : record.target.parentElement;
      if (parent?.closest('[translate="no"]')) continue;
      if (record.type === "characterData") {
        const translated = translateText(record.target.nodeValue || "");
        if (translated !== record.target.nodeValue) record.target.nodeValue = translated;
      } else if (record.type === "attributes") {
        const value = record.target.getAttribute(record.attributeName);
        if (value) {
          const translated = translateText(value);
          if (translated !== value) record.target.setAttribute(record.attributeName, translated);
        }
      } else {
        for (const node of record.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) applyLocale(node);
          else if (node.nodeType === Node.TEXT_NODE) {
            const translated = translateText(node.nodeValue || "");
            if (translated !== node.nodeValue) node.nodeValue = translated;
          }
        }
      }
    }
  });
  observer.observe(document.body, {
    childList: true, subtree: true, characterData: true, attributes: true,
    attributeFilter: ["title", "aria-label", "placeholder"],
  });
}

export { locale };
