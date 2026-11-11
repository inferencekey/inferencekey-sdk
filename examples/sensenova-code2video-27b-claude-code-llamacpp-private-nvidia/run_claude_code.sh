#!/bin/bash
# Drive SenseNova 6.8 Code2Video from Claude Code with the authors' code-video skill.
# Run it in an isolated folder or container: the agent runs shell commands without asking.
# usage: ./run_claude_code.sh /abs/path/to/task  (the folder holds dense_prompt.md)
# needs: Claude Code, Node.js 18+, Python 3, ffmpeg, and `npm install playwright && npx playwright install chromium` in the task folder.
set -euo pipefail
TASK=${1:?task folder}
HERE=$(cd "$(dirname "$0")" && pwd)
set -a; . "$HERE/.env"; set +a
export ANTHROPIC_AUTH_TOKEN="$INFERENCEKEY_API_KEY"   # the InferenceKey API key, never written to the settings file
if [ ! -d "$TASK/.claude/skills/code-video" ]; then
  git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-6.8-Code2Video.git /tmp/sn-c2v
  mkdir -p "$TASK/.claude/skills" && cp -r /tmp/sn-c2v/skills/code-video "$TASK/.claude/skills/"
fi
cd "$TASK"
timeout "${C2V_MINUTES:-60}m" claude -p --settings "$HERE/claude-settings.json" --model sensenova-code2video \
  --max-turns "${C2V_MAX_TURNS:-150}" --permission-mode acceptEdits --allowedTools "Bash,Read,Write,Edit,Glob,Grep,Skill" \
  'Use the project code-video skill. Read .claude/skills/code-video/SKILL.md and dense_prompt.md, then the relevant craft guides. Produce the complete video and deliver out/final.mp4, SPEC.md and VERIFY.md.'
