import os, torch, re, threading, tkinter as tk
from typing import Callable, Optional, Any
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# === Configuration (used only for the optional local backend) ===
BASE_MODEL_PATH = "C:/Users/garag/OneDrive/Desktop/NetAI/mistral_7b_instruct"

# Keep these module-level but DON'T load anything until needed
tokenizer = None
model = None
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

# === Chat history (only used by the local backend) ===
CHAT_HISTORY = []  # (prompt, final_answer) pairs – ONLY these go in history

# ---------- formatting helpers ----------
def clean(txt: str) -> str:
    return re.sub(r"(# ?\d+)+", "", txt).strip()

def inst_first(msg: str) -> str:
    return f"<s>[INST] {msg.strip()} [/INST] "

def inst_later(msg: str) -> str:
    return f"[INST] {msg.strip()} [/INST] "
# === External progress hook (optional, for other scripts) ===
_EXTERNAL_PROGRESS_SINK: Optional[Callable[[str, Any], None]] = None

def register_progress_sink(cb: Optional[Callable[[str, Any], None]]) -> None:
    """
    Other scripts can call register_progress_sink(on_progress) once to receive progress events.
    The callback signature is on_progress(phase: str, payload: Any).

    Phases and payloads (NO model outputs are sent):
      - 'plan'     payload = {'total_steps': int}
      - 'step'     payload = {'index': int, 'total': int, 'status': 'start'|'done'}
      - 'synth'    payload = {'status': 'start'|'done'}
      - 'critique' payload = {'status': 'start'|'done'}
      - 'rewrite'  payload = {'status': 'start'|'done'}      # only emitted if a rewrite happens
      - 'final'    payload = {'status': 'done', 'path': 'draft_ok'|'rewritten'}
    """
    global _EXTERNAL_PROGRESS_SINK
    _EXTERNAL_PROGRESS_SINK = cb


def clear_progress_sink() -> None:
    """Remove any registered external progress sink."""
    register_progress_sink(None)

def _safe_progress_call(cb: Callable, phase: str, payload: Any) -> None:
    """
    Call a progress callback safely. Try (phase, payload); if the callback
    expects no args (legacy), call with no args.
    """
    try:
        cb(phase, payload)
    except TypeError:
        try:
            cb()
        except Exception:
            pass
    except Exception:
        pass

def _emit_progress(local_cb: Optional[Callable], phase: str, payload: Any, sink: Optional[Callable] = None) -> None:
    """
    Emit a progress event to:
      1) the local callback passed into reason(...), if any
      2) the module-level external sink registered via register_progress_sink(...), or an explicit sink
    """
    # Local (per-call) callback
    if callable(local_cb):
        _safe_progress_call(local_cb, phase, payload)

    # External sink (global) or explicitly provided sink
    target = sink if callable(sink) else _EXTERNAL_PROGRESS_SINK
    if callable(target):
        try:
            target(phase, payload)
        except Exception:
            pass

# ---------- local backend (lazy) ----------
def _ensure_local_model_loaded():
    """Load tokenizer/model only if we actually use the local backend."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tok.pad_token = tok.pad_token or tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=quant_cfg,
        device_map="auto",
        offload_state_dict=True,
    ).eval()
    tokenizer, model = tok, mdl

@torch.inference_mode()
def _llm_local(prompt: str, **gen_kwargs) -> str:
    """
    Default backend if none is injected. Uses the same chat formatting scheme
    and a shared CHAT_HISTORY, but loads the model lazily.
    """
    _ensure_local_model_loaded()

    trimmed = CHAT_HISTORY.copy()

    def build_wrapped(pairs):
        segs = []
        for idx, (u, a) in enumerate(pairs):
            wrap = inst_first if idx == 0 else inst_later
            segs.append(f"{wrap(u)}{a.strip()} ")
        segs.append(inst_later(prompt) if pairs else inst_first(prompt))
        return "".join(segs)

    wrapped = build_wrapped(trimmed)
    while len(tokenizer(wrapped, return_tensors="pt")["input_ids"][0]) > 29000 and trimmed:
        trimmed.pop(0)
        wrapped = build_wrapped(trimmed)

    if len(trimmed) < len(CHAT_HISTORY):
        CHAT_HISTORY[:] = trimmed

    inputs = tokenizer(wrapped, return_tensors="pt").to(model.device)
    defaults = dict(max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    outputs  = model.generate(**inputs, **{**defaults, **gen_kwargs})
    prompt_len = inputs["input_ids"].shape[-1]
    new_ids    = outputs[0][prompt_len:]
    return clean(tokenizer.decode(new_ids, skip_special_tokens=True))

# ---------- pass helpers now accept an llm_fn ----------
def get_plan(user_prompt: str, llm_fn: Callable[[str], str]) -> list[str]:
    plan_prompt = (
        "You are an expert planner.\n"
        "Given the user's query, list 3-7 high-level reasoning steps you will follow "
        "to reach a correct response (The number of steps should be directly related to the complexity of the user query). "
        "Number them 1., 2., 3. … on separate lines, one short clause each. "
        "These will be used as your internal reasoning steps because you are a single part of a reasoning system that goes through your list of reasoning steps creating an answer for each, "
        "and then another pass through the model to synthesize the final answer. "
        "You are the one that creates the internal reasoning steps at the beginning, right after the user submits their query. "
        "The user only sees the synthesized final response, not any of the internal responses."
        f"\n\nUser query:\n{user_prompt}"
    )
    plan_text = llm_fn(plan_prompt)
    steps = [clean(s.split(".", 1)[1]) for s in plan_text.splitlines() if re.match(r"\d+\.", s)]
    return [s for s in steps if s]

def run_steps(
    user_prompt: str,
    steps: list[str],
    llm_fn: Callable[[str], str],
    progress_cb=None,
    progress_sink: Optional[Callable[[str, Any], None]] = None,
) -> list[str]:
    """
    Execute each step and optionally:
      - call progress_cb() after each step (GUI progress bar compatibility)
      - emit external progress via progress_sink or the globally-registered sink
        with phase='step' and payload={'index','total','status'}
    """
    thoughts = []
    total = len(steps)
    sink = progress_sink or _EXTERNAL_PROGRESS_SINK

    for idx, step in enumerate(steps, 1):
        # announce start of this inner-thoughts step
        _emit_progress(None, "step", {"index": idx, "total": total, "status": "start"}, sink=sink)

        step_prompt = (
            f"Step {idx}: {step}\n"
            f"Task: Create a paragraph response for this step of the internal reasoning system. You are part of a bigger reasoning system. "
            f"The user will not see your response, your response will only be used internally by AI.\n"
            f"Original user prompt for reference:\n{user_prompt}"
        )
        thoughts.append(llm_fn(step_prompt))

        # announce completion of this step
        _emit_progress(None, "step", {"index": idx, "total": total, "status": "done"}, sink=sink)

        # Local GUI progress (legacy 0-arg callback)
        if progress_cb:
            try:
                progress_cb()
            except Exception:
                pass

    return thoughts



def synthesize(user_prompt: str, steps: list[str], thoughts: list[str], llm_fn: Callable[[str], str]) -> str:
    bullet_plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    merged_thoughts = "\n\n".join(f"### Step {i+1}\n{t}" for i, t in enumerate(thoughts))
    synth_prompt = (
        "Using the internal reasoning data below, write a clear, concise response to the user's query.\n"
        "Cite facts if relevant, omit the scratch-pad.\n\n"
        f"User query:\n{user_prompt}\n\n"
        f"Reasoning Steps (for your eyes only):\n{bullet_plan}\n\n"
        f"Reasoning details (for your eyes only):\n{merged_thoughts}\n\n"
        "This should be an actual synthesized final response. Do not just simply restate the steps data. "
        "Actually give a real synthesized response. Do not split your response into the step numbers, "
        "because the steps are only for your internal reasoning."
    )
    return llm_fn(synth_prompt, max_new_tokens=1024, temperature=0.4)

def critique_draft(user_prompt: str, draft: str, llm_fn: Callable[[str], str]) -> str:
    critique_prompt = (
        "You are a strict reviewer.\n"
        f"User query:\n{user_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "List any factual errors, logic gaps, or unclear parts. Make sure it is actually responding to the most recent user query. Write 'OK' if none."
    )
    return llm_fn(critique_prompt, max_new_tokens=256, temperature=0.3)

def rewrite_draft(user_prompt: str, draft: str, critique: str, llm_fn: Callable[[str], str]) -> str:
    revise_prompt = (
        f"User query:\n{user_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Reviewer notes:\n{critique}\n\n"
        "Rewrite the answer, fixing every issue. Return only the improved answer."
    )
    return llm_fn(revise_prompt, max_new_tokens=1024, temperature=0.4)

def reason(
    user_prompt: str,
    add_to_history: bool = True,
    progress_cb=None,
    *,
    llm_backend: Optional[Callable[..., str]] = None,
) -> str:
    """
    Entry point for external callers.
    - Pass your existing `llm` via llm_backend to reuse the already-loaded model.
    - Progress events contain only stage metadata; no model outputs are sent.
      Phases/payloads:
        plan:     {'total_steps': int}
        step:     {'index': int, 'total': int, 'status': 'start'|'done'}
        synth:    {'status': 'start'|'done'}
        critique: {'status': 'start'|'done'}
        rewrite:  {'status': 'start'|'done'}  # only if rewrite actually runs
        final:    {'status': 'done', 'path': 'draft_ok'|'rewritten'}
    """
    llm_fn = llm_backend if callable(llm_backend) else _llm_local

    # Plan
    steps = get_plan(user_prompt, llm_fn)
    _emit_progress(progress_cb, "plan", {"total_steps": len(steps)})

    # Execute inner-thoughts steps (progress emitted inside run_steps)
    thoughts = run_steps(
        user_prompt,
        steps,
        llm_fn,
        progress_cb=None,         # GUI legacy callback not used here
        progress_sink=progress_cb # send 'step' events to the external listener
    )

    # Synthesize
    _emit_progress(progress_cb, "synth", {"status": "start"})
    draft = synthesize(user_prompt, steps, thoughts, llm_fn)
    _emit_progress(progress_cb, "synth", {"status": "done"})

    # Critique
    _emit_progress(progress_cb, "critique", {"status": "start"})
    critique = critique_draft(user_prompt, draft, llm_fn)
    _emit_progress(progress_cb, "critique", {"status": "done"})

    # Rewrite (if needed)
    if "ok" in (critique or "").lower():
        final = draft
        # no rewrite performed
        _emit_progress(progress_cb, "final", {"status": "done", "path": "draft_ok"})
    else:
        _emit_progress(progress_cb, "rewrite", {"status": "start"})
        final = rewrite_draft(user_prompt, draft, critique, llm_fn)
        _emit_progress(progress_cb, "rewrite", {"status": "done"})
        _emit_progress(progress_cb, "final", {"status": "done", "path": "rewritten"})

    if add_to_history and llm_fn is _llm_local:
        CHAT_HISTORY.append((user_prompt, final))

    return final



# === GUI (Tkinter) ===
class ChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mistral Reasoning Chat")

        # Conversation display
        self.display = ScrolledText(self.root, wrap="word", height=20, width=100, state="disabled")
        self.display.pack(padx=5, pady=5, fill="both", expand=True)

        # Input box
        self.input_box = ScrolledText(self.root, wrap="word", height=6, width=100)
        self.input_box.pack(padx=5, pady=(0, 5), fill="x")

        # Send button
        self.send_btn = tk.Button(self.root, text="Send (Ctrl+Enter)", command=self.on_send)
        self.send_btn.pack(pady=(0, 5))

        # Progress bar (determinate)
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=(0, 5))

        # Bind Ctrl+Enter to send
        self.input_box.bind("<Control-Return>", lambda e: self.on_send())

        # Simple tag colors
        self.display.tag_configure("prompt", foreground="#0080ff")
        self.display.tag_configure("response", foreground="#00aa00")

        # Optional: let GUI use the local backend; you can also inject one if you want:
        self._llm_backend = None  # set to your external llm if desired

    def append_display(self, text: str, tag: str):
        self.display.configure(state="normal")
        self.display.insert("end", text + "\n\n", tag)
        self.display.configure(state="disabled")
        self.display.see("end")

    # ---------- progress helpers ----------
    def prog_reset(self, total: int):
        self.progress["maximum"] = total
        self.progress["value"] = 0

    def prog_inc(self):
        self.progress["value"] += 1

    # ---------- GUI handlers ----------
    def on_send(self):
        user_prompt = self.input_box.get("1.0", "end").strip()
        if not user_prompt:
            return
        self.append_display(f"Prompt:\n{user_prompt}", "prompt")
        self.input_box.delete("1.0", "end")
        self.send_btn.config(state="disabled")
        threading.Thread(target=self.generate_response, args=(user_prompt,)).start()

    def generate_response(self, user_prompt: str):
        # ---- Plan pass ----
        steps = get_plan(user_prompt, self._llm_backend or _llm_local)
        total_expected = 1 + len(steps) + 1 + 1 + 1  # plan + each step + synth + critique + rewrite
        self.root.after(0, lambda: self.prog_reset(total_expected))
        self.root.after(0, self.prog_inc)  # plan done

        # ---- Step passes ----
        thoughts = run_steps(user_prompt, steps, self._llm_backend or _llm_local,
                             progress_cb=lambda: self.root.after(0, self.prog_inc))

        # ---- Synthesis ----
        draft = synthesize(user_prompt, steps, thoughts, self._llm_backend or _llm_local)
        self.root.after(0, self.prog_inc)

        # ---- Critique ----
        critique = critique_draft(user_prompt, draft, self._llm_backend or _llm_local)
        self.root.after(0, self.prog_inc)

        # ---- Rewrite (may or may not advance) ----
        if "ok" in critique.lower():
            final = draft
            self.root.after(0, self.prog_inc)
        else:
            final = rewrite_draft(user_prompt, draft, critique, self._llm_backend or _llm_local)
            self.root.after(0, self.prog_inc)

        # Only store local history if using the local backend
        if (self._llm_backend or _llm_local) is _llm_local:
            CHAT_HISTORY.append((user_prompt, final))

        self.root.after(0, lambda: self.append_display(f"Response:\n{final}", "response"))
        self.root.after(0, self.finish_cycle)

    def finish_cycle(self):
        self.send_btn.config(state="normal")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    ChatGUI().run()
