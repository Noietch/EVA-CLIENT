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
  "collect.vrReviewAria": ["采集数据：左摇杆四向移动、单击下压选择采集位置、双击回放、拨动返回实时，左手 Y 键切换通过或不合格", "Collection data: move with the left stick, press to select a slot, double-press to replay, flick to return live, and use the left-hand Y key to switch pass or fail"],
  "collect.vrReviewHint": ["左摇杆：四向移动、边缘自动翻页 · 单击下压：选择采集位置 · 双击下压：回放 · 回放时拨动：返回实时 · Y：PASS ↔ FAIL", "Left stick: move in four directions and turn pages at the edge · press: select a slot · double-press: replay · flick during replay: return live · Y: PASS ↔ FAIL"],
  "collect.error.selectPosition": ["选择采集位置失败：{error}", "Failed to select collection position: {error}"],
  "collect.error.selectFirst": ["请先用左摇杆选择采集位置", "Use the left stick to select a collection position first"],
  "collect.error.waitSave": ["请等待当前采集或保存完成后再选择采集位置", "Wait for the current capture or save to finish before selecting a collection position"],
  "collect.qc.selectSaved": ["先选择已保存的数据", "Select saved data first"],
  "collect.error.statusSwitch": ["状态切换失败：{error}", "Failed to switch status: {error}"],
  "collect.error.page": ["翻页失败：{error}", "Failed to change page: {error}"],
  "collect.guide.disabled": ["配置中已禁用采集", "Collection disabled in <b>config</b>"],
  "collect.guide.enable": ["开始录制前请启用采集和记录", "Enable collection + logging before recording"],
  "collect.guide.selectTask": ["开始录制前请选择<b>任务</b>", "Select a <b>TASK</b> before recording"],
  "collect.guide.chooseTask": ["请先选择任务", "Choose a task before recording"],
  "collect.guide.motionLocked": ["采集运动已<b>锁定</b>", "Collection motion is <b>locked</b>"],
  "collect.guide.switchArm": ["开始录制前请开启机械臂", "Switch ARM on before START RECORD"],
  "collect.guide.recording": ["录制中 — <b>{frames}</b> 帧", "Recording — <b>{frames}</b> frames"],
  "collect.guide.recordingHint": ["结束/保存会将回合加入队列 · 取消会丢弃当前回合", "END/SAVE queues the episode · CANCEL discards it"],
  "collect.guide.converting": ["转换中 — 队列中有 <b>{count}</b> 项", "Converting — <b>{count}</b> item(s) in queue"],
  "collect.guide.convertingHint": ["请选择绿色项目，然后按回放", "Select a green item, then press REPLAY"],
  "collect.guide.ready": ["准备就绪 — 点击<b>开始录制</b>", "Ready — click <b>START RECORD</b>"],
  "collect.guide.readyHint": ["队列默认折叠；展开查看详情", "Queue is collapsed by default; expand for details"],
  "teleop.vr.linked": ["VR 已连接", "VR LINKED"],
  "teleop.vr.down": ["VR 未连接", "VR DOWN"],
  "teleop.vr.error": ["VR 错误", "VR ERROR"],
  "teleop.input": ["输入设备", "INPUT"],
  "teleop.arm.left.enabled": ["左臂已启用", "LEFT ARM ENABLED"],
  "teleop.arm.left.disabled": ["左臂未启用", "LEFT ARM DISABLED"],
  "teleop.arm.left.unavailable": ["左臂不可用", "LEFT ARM UNAVAILABLE"],
  "teleop.arm.right.enabled": ["右臂已启用", "RIGHT ARM ENABLED"],
  "teleop.arm.right.disabled": ["右臂未启用", "RIGHT ARM DISABLED"],
  "teleop.arm.right.unavailable": ["右臂不可用", "RIGHT ARM UNAVAILABLE"],
  "guide.robotReady": ["机器人就绪", "ROBOT READY"],
  "guide.paused": ["已暂停 — 点击继续", "PAUSED — CLICK RESUME"],
  "guide.setupFailed": ["设置失败", "SETUP FAILED"],
  "guide.awaitingConfig": ["等待配置…", "AWAITING CONFIG…"],
  "guide.preparing": ["自动 · 正在准备机器人…", "AUTO · PREPARING ROBOT…"],
  "guide.rlSelect": ["请选择<b>任务和策略</b>", "Select <b>task and Policy</b>"],
  "guide.rlCriticHint": ["Critic 可选；可用时会增加价值遥测", "Critic is optional and adds value telemetry when available"],
  "guide.rlSetup": ["设置 · <b>{stage}</b>", "SETUP · <b>{stage}</b>"],
  "guide.rlAutoSetup": ["设置将<b>自动</b>开始", "SETUP starts <b>automatically</b>"],
  "guide.rlCriticOptional": ["Critic 可选", "Critic is optional"],
  "guide.rlHilRecording": ["人在回路干预正在<b>录制</b>", "HIL intervention is <b>recording</b>"],
  "guide.rlHilHint": ["接受会继续演练；放弃会回滚当前片段", "ACCEPT resumes rollout; ABANDON rolls the segment back"],
  "guide.rlRunning": ["演练<b>运行中</b>", "Rollout is <b>running</b>"],
  "guide.rlLiveHint": ["Critic 曲线和演练/干预轨迹会实时更新", "The Critic curve and rollout/intervention track update live"],
  "guide.rlSavedReady": ["已保存回合可在<b>回放</b>", "Saved episode is ready for <b>REPLAY</b>"],
  "guide.rlReadyRun": ["准备<b>运行</b>", "Ready to <b>RUN</b>"],
  "guide.rlReplayHint": ["停止、保存，然后选择回合进行回放", "Stop, save, then select an episode to replay"],
  "guide.device": ["设备", "DEVICE"],
  "guide.robotConnected": ["机器人已连接", "ROBOT CONNECTED"],
  "guide.robotDisconnected": ["机器人未连接", "ROBOT DISCONNECTED"],
  "guide.controlEnabled": ["控制已启用", "CONTROL ENABLED"],
  "guide.controlLocked": ["控制已锁定", "CONTROL LOCKED"],
  "guide.follow": ["按左侧步骤 1 → 4 执行", "Follow steps 1 → 4 on the left to run"],
  "guide.selectMode": ["第 2 步 — 在配置中选择<b>模式</b>", "Step 2 — pick a <b>MODE</b> under CONFIG"],
  "guide.selectStrategy": ["第 2 步 — 在配置中选择<b>策略</b>", "Step 2 — pick a <b>STRATEGY</b> under CONFIG"],
  "guide.setupFailedStep": ["第 3 步 — <b>设置失败</b>", "Step 3 — <b>setup failed</b>"],
  "guide.setupFailedHint": ["检查错误，然后点击设置面板中的重试", "Check the error, then click RETRY under SETUP"],
  "guide.setupStage": ["第 3 步 — <b>{stage}</b>…", "Step 3 — <b>{stage}</b>…"],
  "guide.preparingStep": ["第 3 步 — 正在<b>准备机器人</b>…", "Step 3 — <b>preparing the robot</b>…"],
  "guide.autoSetupHint": ["正在自动设置，可能需要几秒钟", "Setting up automatically — this can take a few seconds"],
  "guide.readyRun": ["准备就绪 — 点击<b>运行 ▶</b>开始", "Ready — click <b>RUN ▶</b> to start"],
  "guide.readyRunHint": ["停止以中止 · 重置以回零", "STOP to halt · RESET to home"],
  "guide.running": ["运行中 — 点击<b>停止 ■</b>中止", "Running — click <b>STOP ■</b> to halt"],
  "guide.runningHint": ["右侧显示实时观测", "Live observation on the right"],
  "run.continue": ["继续 ▶▶", "CONTINUE ▶▶"],
  "run.replay": ["回放 ▶", "REPLAY ▶"],
  "run.simPreviewDone": ["仿真预览完成 · {count} 个动作 · 按真实 ▶ 发送", "SIM preview done · {count} actions · press REAL ▶ to dispatch"],
  "run.stepReady": ["准备就绪 · 按仿真 ↻ 推理一个动作块", "READY · press SIM ↻ to infer one chunk"],
  "run.stepIdle": ["空闲 · 请先设置", "IDLE · SETUP FIRST"],
  "run.resume": ["恢复 ▶", "RESUME ▶"],
  "run.hilOn": ["人在回路已开启", "HIL ON"],
  "run.hilOff": ["人在回路已关闭", "HIL OFF"],
  "run.hilUnavailable": ["人在回路不可用", "HIL N/A"],
  "run.hilTitle": ["启用演练中的人在回路干预", "Enable rollout HIL intervention"],
  "device.devices": ["设备", "DEVICES"],
  "device.kind.robot": ["机器人", "Robot"],
  "device.kind.teleop": ["遥操作", "Operation"],
  "device.kind.camera": ["相机", "Camera"],
  "device.real": ["真实", "Real"],
  "device.fake": ["仿真", "Fake"],
  "device.webxr": ["WebXR", "WebXR"],
  "device.openWebxr": ["在头显上打开 WebXR", "Open WebXR on the headset"],
  "device.webxrStreaming": ["WebXR 已在传输", "WebXR is already streaming"],
  "device.killTitle": ["停止全部设备进程并清除设备状态", "Stop every device process and clear device state"],
  "device.killConfirm": ["停止全部会立即强制终止机器人、遥操作和相机进程，并移除机器人力矩。继续吗？", "KILL ALL will force-kill Robot, Operation and Camera processes immediately. Robot torque will be removed. Continue?"],
  "device.serverOutOfDate": ["设备服务版本过旧。命令可能已经发送。请重启 EVA 并刷新页面后再试。", "Device server is out of date. The command may already have been sent. Restart EVA and refresh this page before trying again."],
  "device.selectionTimedOut": ["设备选择超时", "Device selection timed out"],
  "device.commandTimedOut": ["设备命令超时", "Device command timed out"],
  "device.processExited": ["设备进程已退出", "Device process exited"],
  "device.noLocalProcess": ["无本地进程", "No local process"],
  "device.state.stopping": ["停止中", "Stopping"],
  "device.state.failed": ["失败", "Failed"],
  "device.state.running": ["运行中", "Running"],
  "device.state.waiting": ["等待头显", "Waiting for headset"],
  "device.state.starting": ["启动中", "Starting"],
  "device.state.stopped": ["已停止", "Stopped"],
  "device.button.stopping": ["停止中…", "Stopping..."],
  "device.button.stop": ["停止", "Stop"],
  "device.button.cancel": ["取消", "Cancel"],
  "device.button.starting": ["启动中…", "Starting..."],
  "device.button.start": ["启动", "Start"],
  "manual.connectReal": ["连接真实设备", "CONNECT REAL"],
  "manual.disconnect": ["断开连接", "DISCONNECT"],
  "manual.cancel": ["取消", "CANCEL"],
  "manual.stop": ["停止 ■", "STOP ■"],
  "manual.simNoRobot": ["仿真调试 · 无真实机器人（传输类型不是 zmq/ros）", "SIM DEBUG · no real robot (transport not zmq/ros)"],
  "manual.realLive": ["真实机器人 · 实时", "REAL ROBOT · LIVE"],
  "manual.waitingRobot": ["等待机器人…（没有实时数据）", "WAITING FOR ROBOT… (no live data)"],
  "manual.simConnectHint": ["仿真调试 · 按连接真实设备控制硬件", "SIM DEBUG · press CONNECT REAL to drive hardware"],
  "manual.publishRate": ["发布频率 Hz", "publish rate Hz"],
  "manual.targetCurrentQpos": ["目标 / 当前 QPOS", "TARGET / CURRENT QPOS"],
  "manual.targetCurrent": ["目标 / 当前", "target / current"],
  "manual.enableTeleop": ["启用遥操作运动", "Enable teleop motion"],
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

const locale = globalThis.document?.documentElement?.dataset?.evaLanguage === "zh" ? "zh" : "en";
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
  const elements = [];
  if (root?.nodeType === 1 && root.matches("[title], [aria-label], [placeholder]")) {
    elements.push(root);
  }
  if (root.querySelectorAll) elements.push(...root.querySelectorAll("*"));
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

function explicitElements(root, selector) {
  const elements = [];
  if (root?.nodeType === 1 && root.matches(selector)) elements.push(root);
  if (root.querySelectorAll) elements.push(...root.querySelectorAll(selector));
  return elements;
}

function applyExplicitTranslations(root) {
  for (const element of explicitElements(root, "[data-i18n]")) {
    element.textContent = t(element.dataset.i18n);
  }
  for (const element of explicitElements(root, "[data-i18n-html]")) {
    element.innerHTML = t(element.dataset.i18nHtml);
  }
  for (const [attribute, datasetKey] of [
    ["title", "i18nTitle"],
    ["aria-label", "i18nAriaLabel"],
    ["placeholder", "i18nPlaceholder"],
  ]) {
    for (const element of explicitElements(root, `[data-i18n-${attribute}]`)) {
      element.setAttribute(attribute, t(element.dataset[datasetKey]));
    }
  }
}

export function applyLocale(root = document) {
  translateNode(root);
  translateAttributes(root);
  applyExplicitTranslations(root);
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
