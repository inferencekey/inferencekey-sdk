# Prompts and tool schemas from the NorthCUA Harness

`computer_using_agent_*.j2` and `desktop_tools.json` come from
[beipo-ai/NorthCUA](https://github.com/beipo-ai/NorthCUA) at commit
`b904a8fc4fbc16de0197a142f6dd9ab2bd36ffff` (Apache-2.0, Copyright the NorthCUA
authors): the Desktop agent prompt templates
(`northcua/agents/desktop/prompt/templates/`) unchanged, and the ten desktop
action tool schemas generated from
`northcua_runtime_shared/desktop_surface/action_contracts.py`
(`model_json_schema()` of each parameter model, in OpenAI function format).
