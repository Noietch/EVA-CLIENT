## Agent Behavior and Communication Rules

> Bias toward caution over speed. Trivial tasks may use judgment. Working signal: fewer unneeded diff lines, fewer rewrites from overcomplication, clarifying questions come *before* implementation, not after.

### Basic Interaction
- **Language**: Think in English internally; answer and explain in Chinese.
- **Think Before Coding**: State assumptions; if multiple interpretations exist, surface them rather than pick silently; if unclear, stop and ask; if simpler exists, say so and push back.
- **Goal-Driven Execution**: Before coding, define a verifiable success criterion (e.g. "add validation" → write failing tests for invalid input, then make pass; "fix bug" → write reproducer, then make pass). For multi-step work, state a brief `step → verify` plan and loop until each verify passes.
- **No Unapproved Commits**: Never trigger `git commit` automatically.
- **No AI Signature**: Do not include tags like `coauthor: claude code` in any output.

### Coding Rules
- **Simplicity First**: Minimum code that solves the problem; nothing speculative. No unrequested features / abstractions / flexibility, no error handling for impossible cases. If 200 lines can be 50, rewrite. Self-test: would a senior engineer call it overcomplicated?
- **Surgical Changes**: Touch only what the request requires. Don't "improve" adjacent code, comments, or formatting; don't refactor things that aren't broken; match existing style. Remove imports / vars / funcs *your* change orphaned; don't delete pre-existing dead code unless asked — mention it in chat instead. Every changed line must trace directly to the request.
- **Defensive Programming Limits**: Avoid unnecessary `try-except` and safety branching. Write the main path assuming inputs are valid.
- **Dependency Management**: All imports at the top of the file. Never modify third-party / external module source.
- **kwargs**: Avoid `kwargs.get`. Declare needed variables directly.
- **Function Design**: Prefer keeping related logic in one function over splitting into many small ones; reduce jumps.
- **Function Docstring**: Required for public, non-trivial functions (tensor ops, model forward, data transforms, anything with shape-carrying args). Triple-quoted: one-line summary, then `Args:` / `Returns:` with tensor shapes / types. Skip for small helpers whose name + signature are obvious; no boilerplate padding.
- **Comment Policy**: All comments in English. Default to none. Add a short `#` only when (a) labeling stages of a tensor-shape pipeline, or (b) the *why* is non-obvious (hidden constraint, workaround, surprising invariant). Don't label ordinary control flow / config / IO — names suffice. **Never** narrate the edit ("added for X", "fix Y", "see issue Z"); that belongs in the commit message.
- **Comment Placement**: Put a comment at the *start* of the block it describes or at the *end of the annotated line* — never wedged between consecutive statements of one sequence, and never inside a literal (dict / list / call args). E.g. `x = 0  # why` (good), or a leading line above the block (good); not a `#` line between two dict keys (bad).
- **Virtual Environment Dependencies**: Reflect every virtual-environment dependency change in `pyproject.toml`.
- **Comment Policy** — example of when comments DO help:
  ```python
  def flatten_views(x):
      """Flatten multi-view video tensor into a single batch dim and move channel to last.

      Args:
          x: [B, V, T, C, H, W] or [B, T, C, H, W]

      Returns:
          out: [B*V, T, H, W, C]  (channel-last)
          meta: (B, V)
      """
      # normalize to 6D by inserting a view dim when missing
      if x.ndim == 5:
          x = x.unsqueeze(1)  # [B, 1, T, C, H, W]
      B, V = x.shape[:2]

      # merge batch & view, move channel to last
      x = x.flatten(0, 1)           # [B*V, T, C, H, W]
      out = x.permute(0, 1, 3, 4, 2)  # [B*V, T, H, W, C]
      return out, (B, V)
  ```

- **Virtual Environment**
  All operations must be executed inside the designated virtual environment:
```bash
source .venv/bin/activate
```

- **Network Proxy Management**
  The network proxy is only for pulling code or external resources. Before starting any training job, the proxy must be disabled with `unset`.
```bash
# Enable proxy
# proxyon
proxy_on() {
    export http_proxy="http://127.0.0.1:7897"
    export https_proxy="http://127.0.0.1:7897"
    export HTTP_PROXY="http://127.0.0.1:7897"
    export HTTPS_PROXY="http://127.0.0.1:7897"
    echo "HTTP/HTTPS Proxy on"
}

# proxyoff
proxy_off() {
    unset http_proxy
    unset https_proxy
    echo "HTTP/HTTPS Proxy off"
}
```
