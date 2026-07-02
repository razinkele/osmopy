# Claude memory mirror

This directory is a **mirror** of the file-based Claude memory store that lives outside the repo at:

    ~/.claude/projects/-home-razinka-osmose/memory/

`MEMORY.md` is the index (one line per memory); the other `*.md` files are individual
memories (frontmatter + one fact each). This copy exists so the memory store is captured under
version control alongside the code it describes.

**Source of truth is the `~/.claude/...` store, not this mirror.** Re-sync with:

    cp -a ~/.claude/projects/-home-razinka-osmose/memory/. docs/claude-memory/

Do not hand-edit files here expecting them to flow back — edits belong in the live store.
