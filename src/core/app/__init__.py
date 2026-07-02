"""Application engine — the main loop, command handlers, runtime/session state,
and the web console that serves as the sole operator interface.

Modules:
    run       main loop + web command dispatch + run() entry point
    handlers  action handlers (policy connect, publish, setup/reset/run)
    state     SessionState / RuntimeState dataclasses and helpers
    console   stdlib HTTP console server (browser → command_queue)
"""
