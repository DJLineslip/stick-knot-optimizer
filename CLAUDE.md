@AGENTS.md

## Claude Code notes

- Start long runs as background tasks that log to results/logs/, then poll the logs; do not block the session on commands that take more than a few minutes.
- Use plan mode before changing anything in equistick/geometry.py or equistick/certify.py.
