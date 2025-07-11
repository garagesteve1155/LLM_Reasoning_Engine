import os, torch, re, threading, tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === Configuration ===
BASE_MODEL_PATH = "C:/Complete/Path/To/Model/Folder"

# === CUDA + model init ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quant_cfg,
    device_map="auto",
    offload_state_dict=True,
).eval()

# === Helpers ===
CHAT_HISTORY = []  # (prompt, final_answer) pairs – ONLY these go in history

def clean(txt: str) -> str:
    return re.sub(r"(# ?\d+)+", "", txt).strip()

def inst_first(msg: str) -> str:
    return f"<s>[INST] {msg.strip()} [/INST] "

def inst_later(msg: str) -> str:
    return f"[INST] {msg.strip()} [/INST] "

@torch.inference_mode()
def llm(prompt: str, **gen_kwargs) -> str:
    """Generate with proper v0.3 chat formatting (single <s>) and
       trim oldest history until ≤ 29000 tokens."""
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


# ---------- Pass 1: PLAN ----------
def get_plan(user_prompt: str) -> list[str]:
    plan_prompt = (
        "You are an expert planner.\n"
        "Given the user's query, list 3-7 high-level reasoning steps you will follow "
        "to reach a correct response (The number of steps should be directly related to the complexity of the user query). Number them 1., 2., 3. … on separate lines, one short clause each. These will be used as your internal reasoning steps because you are a single part of a reasoning system that goes through your list of reasoning steps creating an answer for each, and then another pass through the model to synthesize the final answer. You are the one that creates the internal reasoning steps at the beginning, right after the user submits their query. The user only sees the synthesized final response, not any of the internal responses."
        f"\n\nUser query:\n{user_prompt}"
    )
    plan_text = llm(plan_prompt)
    steps = [clean(s.split(".", 1)[1]) for s in plan_text.splitlines() if re.match(r"\d+\.", s)]
    return [s for s in steps if s]

# ---------- Pass 2: EXECUTE EACH STEP ----------
def run_steps(user_prompt: str, steps: list[str], progress_cb=None) -> list[str]:
    thoughts = []
    for idx, step in enumerate(steps, 1):
        step_prompt = (
            f"Step {idx}: {step}\n"
            f"Task: Create a paragraph response for this step of the internal reasoning system. You are part of a bigger reasoning system. The user will not see your response, your response will only be used internally by AI.\n"
            f"Original user prompt for reference:\n{user_prompt}"
        )
        thoughts.append(llm(step_prompt))
        if progress_cb:
            progress_cb()
    return thoughts

# ---------- Pass 3: FINAL SYNTHESIS ----------
def synthesize(user_prompt: str, steps: list[str], thoughts: list[str]) -> str:
    bullet_plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    merged_thoughts = "\n\n".join(f"### Step {i+1}\n{t}" for i, t in enumerate(thoughts))
    synth_prompt = (
        "Using the internal reasoning data below, write a clear, concise response to the user's query.\n"
        "Cite facts if relevant, omit the scratch-pad.\n\n"
        f"User query:\n{user_prompt}\n\n"
        f"Reasoning Steps (for your eyes only):\n{bullet_plan}\n\n"
        f"Reasoning details (for your eyes only):\n{merged_thoughts}\n\n"
        "This should be an actual synthesized final response. Do not just simply restate the steps data. Actually give a real synthesized response. Do not split your response into the step numbers, because the steps are only for your internal reasoning."
    )
    return llm(synth_prompt, max_new_tokens=1024, temperature=0.4)

# ---------- Pass 4: CRITIQUE ----------
def critique_draft(user_prompt: str, draft: str) -> str:
    critique_prompt = (
        "You are a strict reviewer.\n"
        f"User query:\n{user_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "List any factual errors, logic gaps, or unclear parts. Make sure it is actually responding to the most recent user query. Write 'OK' if none."
    )
    return llm(critique_prompt, max_new_tokens=256, temperature=0.3)

# ---------- Pass 5: REWRITE ----------
def rewrite_draft(user_prompt: str, draft: str, critique: str) -> str:
    revise_prompt = (
        f"User query:\n{user_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Reviewer notes:\n{critique}\n\n"
        "Rewrite the answer, fixing every issue. Return only the improved answer."
    )
    return llm(revise_prompt, max_new_tokens=1024, temperature=0.4)

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
        steps = get_plan(user_prompt)
        total_expected = 1 + len(steps) + 1 + 1 + 1  # plan + each step + synth + critique + rewrite
        self.root.after(0, lambda: self.prog_reset(total_expected))
        self.root.after(0, self.prog_inc)  # plan done

        # ---- Step passes ----
        thoughts = run_steps(user_prompt, steps, progress_cb=lambda: self.root.after(0, self.prog_inc))

        # ---- Synthesis ----
        draft = synthesize(user_prompt, steps, thoughts)
        self.root.after(0, self.prog_inc)

        # ---- Critique ----
        critique = critique_draft(user_prompt, draft)
        self.root.after(0, self.prog_inc)

        # ---- Rewrite (may or may not advance) ----
        if "ok" in critique.lower():
            final = draft
            self.root.after(0, self.prog_inc)
        else:
            final = rewrite_draft(user_prompt, draft, critique)
            self.root.after(0, self.prog_inc)

        CHAT_HISTORY.append((user_prompt, final))
        self.root.after(0, lambda: self.append_display(f"Response:\n{final}", "response"))
        self.root.after(0, self.finish_cycle)

    def finish_cycle(self):
        self.send_btn.config(state="normal")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    ChatGUI().run()
