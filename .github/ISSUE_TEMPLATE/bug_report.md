---
name: Bug report
about: Something isn't working as expected
labels: bug
---

**Which layer?**
- [ ] Observer hook (UserPromptSubmit — registers uncertain values)
- [ ] Gate hook (PreToolUse — blocks writes)
- [ ] Scan annotations (code output annotations)
- [ ] Rust gate (`credence-gate` binary)
- [ ] Cross-session memory (snapshot / recall)
- [ ] Registry / decay
- [ ] MCP server
- [ ] Other

**What happened?**
Describe the behavior you saw.

**What did you expect?**
What should have happened instead.

**Minimal reproduction**
```python
# Paste the smallest code that triggers the bug
```

**The uncertain constraint text** (if applicable)
```
# Paste the exact text that should have triggered / not triggered the probe
```

**Environment**
- OS:
- Python version:
- `pip show credence-enforce` output:
- Rust gate built? (yes/no):

**Offline or API?**
- [ ] Offline (no API call involved)
- [ ] API call involved (model: ___ )
