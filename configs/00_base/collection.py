"""Shared recording schema for workflows that collect demonstrations."""

collection = dict(
    schema=dict(
        columns=dict(
            qpos="observations.state.qpos",
            eef="observations.state.eef",
            action_qpos="action.qpos",
            action_eef="action.eef",
        ),
    ),
)
