"""Canonical command metadata shared by the web console and control channel."""

from __future__ import annotations

_COMMANDS = (
    ("bootstrap", "web:bootstrap", ()),
    ("warmup", "web:warmup", ("be-setup",)),
    ("switch_task", "web:switch_task:{task}", ("prompt-list",)),
    ("rl_select_task", "web:rl_select_task:{task}", ("rl-task-list",)),
    ("rl_select_policy", "web:rl_select_policy:{slot}", ("rl-policy-list",)),
    ("rl_select_critic", "web:rl_select_critic:{slot}", ("rl-critic-list",)),
    ("rl_setup", "web:rl_setup", ("rl-b-setup",)),
    ("start", "web:start", ("be-run",)),
    ("stop", "web:stop", ()),
    ("run", "web:run", ("b-run", "b-replay-run", "rl-b-run")),
    ("halt", "web:halt", ("b-step-halt", "b-replay-step-halt", "rl-b-intervene")),
    ("reset", "web:reset", ("be-reset",)),
    (
        "console_reset",
        "web:console_reset",
        ("b-reset", "b-step-reset", "b-replay-reset", "b-replay-step-reset", "rl-b-reset"),
    ),
    ("init_move", "web:init_move:{qpos}", ()),
    ("init_gripper", "web:init_gripper:{action}:{side}", ()),
    ("init_done", "web:init_done", ()),
    ("switch_ckpt", "web:switch_ckpt:{slot}", ()),
    ("operator_action", "web:operator_action:{intent}:{source}", ("b-collect-toggle",)),
    ("connect", "web:connect", ("bm-connect",)),
    ("disconnect", "web:disconnect", ("bm-connect",)),
    ("select_mode", "web:select_mode:{mode}", ("mode-list",)),
    ("select_episode", "web:select_episode:{episode}", ()),
    ("load_replay_dataset", "web:load_replay_dataset:{payload}", ()),
    ("clear_replay", "web:clear_replay", ()),
    ("set_replay_fps", "web:set_replay_fps:{fps}", ("replay-tune-fps",)),
    ("replay_seek", "web:replay_seek:{frame}", ("scrub-range",)),
    ("select_strategy", "web:select_strategy:{strategy}", ("strategy-list",)),
    ("update_infer_params", "web:update_infer_params:{payload}", ()),
    ("rollout_intervention_abandon", "web:rollout_intervention_abandon", ("rl-b-abandon",)),
    (
        "rollout_intervention_enabled",
        "web:rollout_intervention_enabled:{enabled}",
        ("rl-hil-enable",),
    ),
    ("rollout_save", "web:rollout_save", ("rl-b-save",)),
    ("rollout_stop", "web:rollout_stop", ()),
    ("tab_switch", "web:tab_switch:{tab}", ()),
    ("step_infer", "web:step_infer", ("b-step", "b-replay-step")),
    ("step_commit", "web:step_commit", ("b-commit", "b-replay-commit")),
    ("step_cancel", "web:step_cancel", ("b-step-halt", "b-replay-step-halt")),
    ("manual_qpos", "web:manual_qpos:{qpos}", ()),
    ("manual_dispatch", "web:manual_dispatch:{dispatch}", ("bm-send",)),
    ("manual_send", "web:manual_send", ("bm-send",)),
    ("manual_home", "web:manual_home", ("bm-home",)),
    ("collect_start", "web:collect_start", ()),
    ("collect_stop", "web:collect_stop", ()),
    ("collect_cancel", "web:collect_cancel", ("b-collect-cancel",)),
    ("select_collect_task", "web:select_collect_task:{task}", ("collect-prompt-list",)),
    (
        "gripper",
        "web:gripper:{side}:{state}:{lock}",
        ("gripper-buttons", "eval-gripper-buttons", "rl-gripper-buttons"),
    ),
)

WEB_COMMAND_VERBS = frozenset(entry[0] for entry in _COMMANDS)


def control_command_catalog() -> list[dict[str, object]]:
    """Return JSON-ready command metadata for external shortcut clients."""
    return [
        {"verb": verb, "command": command, "controls": list(controls)}
        for verb, command, controls in _COMMANDS
    ]
