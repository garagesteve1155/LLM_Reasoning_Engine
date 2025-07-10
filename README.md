A reasoning system for Mistral 7b Instruct v0.3

Small desktop app (Tkinter) that runs a single **Mistral-7B** model
locally and adds a multi-pass reasoning loop:

1. **Plan** – model generates numbered high-level steps  
2. **Execute** – creates a hidden paragraph for each step  
3. **Synthesize** – merges internal thoughts into a draft answer  
4. **Critique** – self-reviewer lists flaws, or “OK”  
5. **Rewrite** – fixes draft if needed

The progress bar reflects each internal pass so you know the model
hasn’t frozen.  Prompts and responses are labelled in the chat pane.

## Quick start
```bash
git clone https://github.com/garagesteve1155/mistral-reasoning-gui
cd mistral-reasoning-gui
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python chat_reasoning_gui.py
