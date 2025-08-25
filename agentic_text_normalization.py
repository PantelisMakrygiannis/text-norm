"""
Agentic Text Normalization System (refactored, single-file)

- Complexity analysis via a small Hugging Face LLM with a lightweight prompt (fallback to heuristics).
- Strategy selection maps complexity -> low|medium|high extraction pipelines.
- Low complexity: remove known non-writer entities (publishers, orgs) + simple cleanup.
- Medium complexity: NER (spaCy) + regex-based person / stage-name extraction.
- High complexity: LLM-based normalization with a few-shot prompt, fallback to Medium if uncertain.
- Formatter agent: only formatting (comma inversion, casing, de-duplication, joining with "/"). No entity removal.
- Orchestrator agent: coordinates, picks strategy via mapping.

Run:
  pip install -r requirements.txt
  python agentic_text_normalization.py --input "Smith, John/Jane Doe"

Demo:
  python agentic_text_normalization.py --demo

Evaluate on CSV (expects columns: raw_comp_writers_text, CLEAN_TEXT):
  python agentic_text_normalization.py --eval_csv data.csv --sample 500

Notes:
- Dependencies degrade gracefully (spaCy model / transformers optional).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Centralized Utilities
# -----------------------------
SEPARATORS_PATTERN = re.compile(r"\s*(?:/|,|&| and |;|\||\\)\s*", re.IGNORECASE)
TRIM_PARENS = re.compile(r"\((.*?)\)")
MULTISPACE = re.compile(r"\s{2,}")

UNKNOWN_MARKERS = [
    "<unknown>", "unknown writer", "unknown", "n/a", "none", "-", "(999990)", "(000000)", "999990"
]

KNOWN_NON_WRITER_ENTITIES = [
    # Common publishing/company terms and rights statements
    r"copyright control", r"publishing", r"music(?!\s*by)", r"records?", r"recordings?", r"rights?", r"licen[cs]e",
    r"limited", r"ltd\.?", r"llc", r"inc\.?", r"corp\.?", r"company", r"co\.?", r"bv", r"gmbh", r"sa\b", r"sas\b",
    r"universal", r"sony", r"warner", r"bmg", r"emi", r"atv", r"ascap", r"bmi", r"sesac", r"prs", r"socan", r"sacem",
    r"apra", r"amra", r"koda", r"sabam", r"zaiks", r"kobalt", r"concord", r"downtown", r"peer", r"casa",
    r"administration", r"admin", r"sub.?pub", r"production[s]", r"publisher", r"publishers", r"edition",
    r"featuring", r"producer", r"arranger", r"(c)\b", r"\u00a9"
]
# Extra entity names explicitly removed in low complexity (user examples)
EXPLICIT_REMOVE = [r"ibm", r"sony", r"atv", r"publishers?", r"multiple", r"/ publishers", r"/publisher"]

PERSON_NAME_PATTERNS = [
    re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$'),                  # First Last
    re.compile(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+$'),                 # Last, First
    re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$'),    # First Middle Last
    re.compile(r'^[A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+$'),       # First M. Last
    re.compile(r'^[A-Z]\.?\s*[A-Z][a-z]+\s+[A-Z][a-z]+$'),       # F. Middle Last
    re.compile(r'^[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]+$'),      # International names
]

NOISE_COMPLEXITY_PATTERNS = [
    re.compile(r'&[^/]*&'),
    re.compile(r'[,;]{2,}'),
    re.compile(r'[A-Z]{3,}\s+[A-Z]{3,}'),
    re.compile(r'\([^)]*\)'),
    re.compile(r'[^\x00-\x7F]'),
]

COMMA_INVERTED = re.compile(r"^\s*([A-Za-z'`\-]+)\s*,\s*([A-Za-z].*?)\s*$")

def split_candidates(text: str) -> List[str]:
    return [p.strip() for p in SEPARATORS_PATTERN.split(text) if p.strip()]

def normalize_spaces(s: str) -> str:
    s = MULTISPACE.sub(" ", s.strip())
    s = re.sub(r"\s*/\s*", "/", s)
    s = MULTISPACE.sub(" ", s.strip())
    return s

def is_unknown(token: str) -> bool:
    t = token.strip().lower()
    return any(mark in t for mark in UNKNOWN_MARKERS)

def build_non_writer_regex() -> re.Pattern:
    word_group = "|".join(KNOWN_NON_WRITER_ENTITIES + EXPLICIT_REMOVE)
    return re.compile(rf"\b(?:{word_group})\b", re.IGNORECASE)

def preserve_hyphenated_capitalization(token: str) -> bool:
    return bool(re.match(r"^[A-Z][a-z]+(?:-[A-Z][a-z]+)+(?: [A-Z][a-z]+)*$", token))

# -----------------------------
# Base Agent Interfaces
# -----------------------------
class Agent:
    name: str = "agent"
    def run(self, *args, **kwargs):
        raise NotImplementedError

class LLMAgent(Agent):
    """Base agent for HuggingFace text2text LLMs."""
    def __init__(self, model: str, device: Optional[str] = None):
        self.model_id = model
        self.device = device
        self.pipe = self._init_llm()

    def _init_llm(self):
        try:
            from transformers import pipeline
            return pipeline("text2text-generation", model=self.model_id, device=self.device)
        except Exception:
            return None

# -----------------------------
# Complexity Analysis Agent
# -----------------------------
@dataclass
class ComplexityOutput:
    label: str  # LOW | MEDIUM | HIGH
    confidence: float
    rationale: Optional[str] = None

class ComplexityAgent(LLMAgent):
    name = "complexity_agent"

    def __init__(self, model: str = "google/flan-t5-large", device: Optional[str] = None):
        super().__init__(model=model, device=device)

    @staticmethod
    def _heuristic_complexity(text: str) -> ComplexityOutput:
        t = text.lower()
        score = 0
        # weighted features
        if len(split_candidates(text)) > 4:
            score += 1
        if "(" in text or ")" in text:
            score += 1
        if "," in text:
            score += 1
        if any(p.search(text) for p in NOISE_COMPLEXITY_PATTERNS):
            score += 1
        if any(k in t for k in ["artist", "feat", "featuring"]):
            score += 1
        if any(k in t for k in ["copyright", "publishing", "control", "llc", "inc", "ltd", "music"]):
            score += 1
        if re.search(r"[^\x00-\x7F]", text):
            score += 1
        if re.search(r"\b[A-Z]{3,}\b", text):
            score += 1

        if score <= 1:
            label = "LOW"
        elif 2 <= score <= 3:
            label = "MEDIUM"
        else:
            label = "HIGH"
        return ComplexityOutput(label=label, confidence=0.6, rationale=f"heuristic score={score}")

    def run(self, text: str) -> ComplexityOutput:
        if not text or not text.strip():
            return ComplexityOutput(label="LOW", confidence=1.0, rationale="empty")

        # If LLM is unavailable, or exceptions happen -> heuristic
        if self.pipe is None:
            return self._heuristic_complexity(text)

        prompt = f"""
        You classify text normalization difficulty for music writer strings. 
        Output exactly one JSON object with keys: label, reason. 
        Labels: 
        - LOW = simple removal of known orgs & cleanup 
        - MEDIUM = needs NER/person detection and comma inversions 
        - HIGH = requires LLM reasoning (nested parentheses, messy separators, tricky cases).

        Examples:

        Input: "John Smith"
        Output: {{"label": "LOW", "reason": "Single clean name"}}

        Input: "Smith, John/Jane Doe"
        Output: {{"label": "MEDIUM", "reason": "Comma inversion + multiple names"}}

        Input: "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)"
        Output: {{"label": "HIGH", "reason": "Nested parentheses + multiple separators"}}

        Now classify:
        Input: "{text}"
        Output:
            """.strip()
        
        try:
            out = self.pipe(prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]
            m = re.search(r"\{.*\}", out, re.DOTALL)
            if m:
                import json as _json
                js = _json.loads(m.group(0))
                label = js.get("label", "MEDIUM").upper()
                reason = js.get("reason", "llm")
            else:
                m2 = re.search(r"(LOW|MEDIUM|HIGH)", out.upper())
                label = m2.group(1) if m2 else "MEDIUM"
                reason = out.strip()

            if label == "LOW" and ("," in text and "(" in text or ("&" in text)):
                return self._heuristic_complexity(text)
            
            return ComplexityOutput(label=label, confidence=0.7, rationale=reason)
        except Exception:
            return self._heuristic_complexity(text)

# -----------------------------
# Strategy Selector Agent
# -----------------------------
class StrategySelectorAgent(Agent):
    name = "strategy_selector"
    def run(self, complexity: ComplexityOutput) -> str:
        return {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}.get(complexity.label.upper(), "medium")

# -----------------------------
# Low Complexity Agent
# -----------------------------
class LowComplexityCleanerAgent(Agent):
    name = "low_cleaner"

    def __init__(self):
        self.non_writer_re = build_non_writer_regex()

    def _clean_token(self, token: str) -> Optional[str]:
        token = TRIM_PARENS.sub("", token)  # strip parens content

        if self.non_writer_re.search(token):
            return None  # discard non-writer entities completel
        # token = self.non_writer_re.sub("", token)  # strip orgs/publishers
        token = normalize_spaces(token)
        if not token or is_unknown(token):
            return None
        if preserve_hyphenated_capitalization(token):
            return token
        return token

    def run(self, text: str) -> List[str]:
        parts = split_candidates(text)
        out: List[str] = []
        for p in parts:
            ct = self._clean_token(p)
            if ct:
                out.append(ct)
        return out

# -----------------------------
# Medium Complexity Agent: spaCy NER + regex
# -----------------------------
class MediumComplexityNERAgent(Agent):
    name = "medium_ner"

    def __init__(self):

        self.nlp = None
        try:
            import spacy
            # Try to load a small English model; if not present, fallback to blank with NER disabled
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = spacy.blank("en")
        except Exception:
            self.nlp = None


        self.stage_name_indicators = ['DJ', 'MC', 'LIL', 'YOUNG', 'BIG', 'KING', 'QUEEN', 'SIR', 'LADY']

    def _looks_like_person_name(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 2 or len(text) > 50:
            return False
        return any(pat.match(text) for pat in PERSON_NAME_PATTERNS)

    def _looks_like_stage_name(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        tu = text.upper()
        for ind in self.stage_name_indicators:
            if tu.startswith(ind + ' ') or tu == ind or tu.startswith(ind):
                return True
        single = [
            r'^[A-Z][a-z]{2,15}$',  # single capitalized word
            r'^[A-Z]{2,16}$',       # all caps alias
            r'^[A-Z][a-z]+[0-9]+$', # letters+digits
        ]
        if any(re.match(p, text) for p in single):
            if tu not in {'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM'}:
                return True
        return False

    @staticmethod
    def _fix_inversion(name: str) -> str:
        m = COMMA_INVERTED.match(name)
        if m:
            last, first = m.groups()
            return normalize_spaces(f"{first} {last}")
        return name

    def run(self, text: str) -> List[str]:
        candidates = split_candidates(text)
        persons: List[str] = []

        # Run spaCy if available; otherwise fall back to heuristics
        if self.nlp is not None and hasattr(self.nlp, "pipe"):
            try:
                print("Running SpaCy NER on candidates:", candidates)
                docs = list(self.nlp.pipe(candidates))
                for part, doc in zip(candidates, docs):
                    ents = [(ent.text, ent.label_) for ent in doc.ents]
                    print(f"Input: '{part}' → SpaCy Entities Detected: {ents}")

                    found = [ent.text for ent in getattr(doc, "ents", []) if ent.label_ == "PERSON"]
                    if found:
                        for f in found:
                            persons.append(self._fix_inversion(f))
                    else:
                        if self._looks_like_person_name(part) or self._looks_like_stage_name(part):
                            persons.append(self._fix_inversion(part))
            except Exception:
                # fallback if spaCy threw, apply regex heuristics
                for part in candidates:
                    if self._looks_like_person_name(part) or self._looks_like_stage_name(part):
                        persons.append(self._fix_inversion(part))
        else:
            for part in candidates:
                if self._looks_like_person_name(part) or self._looks_like_stage_name(part):
                    persons.append(self._fix_inversion(part))

        # Deduplicate preserving order
        seen = set()
        out: List[str] = []
        for p in persons:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

# -----------------------------
# High Complexity Agent: LLM with few-shot, fallback to Medium
# -----------------------------
class HighComplexityLLMAgent(LLMAgent):
    name = "high_llm"

    def __init__(self, model: str = "google/flan-t5-large", device: Optional[str] = None):
        super().__init__(model=model, device=device)
        self.fallback_medium = MediumComplexityNERAgent()

    def run(self, text: str) -> List[str]:
        if self.pipe is None:
            return self.fallback_medium.run(text)

        prompt = (
            "You normalize music writer metadata.\n"
            "Rules:\n"
            "- Extract ALL human names, including inside parentheses.\n"
            "- Convert 'Last, First' -> 'First Last'.\n"
            "- Merge descriptors into the name (e.g., 'Complex (Artist)' -> 'Complex Artist').\n"
            "- Remove publishers/companies/orgs/unknown markers.\n"
            "- Return ONLY names, separated by slashes (/). No extra text.\n"
            "- Do not invent names.\n\n"
            "Examples:\n"
            "Input: Smith, John/Jane Doe\nOutput: John Smith/Jane Doe\n"
            "Input: Day & Murray (Bowles, Gaudet, Middleton & Shanahan)\nOutput: Day/Murray/Bowles/Gaudet/Middleton/Shanahan\n"
            "Input: Complex (Artist) & Multiple / Publishers, LLC\nOutput: Complex Artist\n"
            "Input: UNKNOWN WRITER (999990)\nOutput:\n"
            "Input: John Smith & Co. / Big Music Publishing\nOutput: John Smith\n\n"
            f"Input: {text}\nOutput:"
        )
        try:
            out = self.pipe(prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]
            guessed = [p.strip() for p in SEPARATORS_PATTERN.split(out) if p.strip()]
            if not guessed and out.strip():
                guessed = [out.strip()]
            if not guessed:
                return self.fallback_medium.run(text)
            return guessed
        except Exception:
            return self.fallback_medium.run(text)

# -----------------------------
# Formatter Agent (formatting only)
# -----------------------------
class FormatterAgent(Agent):
    name = "formatter"

    @staticmethod
    def _fix_comma_inversion(name: str) -> str:
        m = COMMA_INVERTED.match(name)
        if m:
            last, first = m.groups()
            return normalize_spaces(f"{first} {last}")
        return name

    def _format_one(self, name: str) -> str:
        if not name:
            return ""
        name = TRIM_PARENS.sub("", name)          # strip leftover parens content (just in case)
        name = normalize_spaces(name)
        name = self._fix_comma_inversion(name)
        return name
    
    def run(self, names: List[str]) -> str:
        # Only formatting: normalize spaces, fix commas, casing, dedupe, and join with "/"
        cleaned: List[str] = []
        seen: set = set()
        for n in names:
            if not n or is_unknown(n):
                continue
            n = self._format_one(n)
            if not n:
                continue
            key = n.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(n)
        return "/".join(cleaned)

# -----------------------------
# Orchestrator Agent
# -----------------------------
class OrchestratorAgent(Agent):
    name = "orchestrator"

    def __init__(self, llm_model: str = "google/flan-t5-large", device: Optional[str] = None):
        self.complexity = ComplexityAgent(model=llm_model, device=device)
        self.selector = StrategySelectorAgent()
        self.low = LowComplexityCleanerAgent()
        self.medium = MediumComplexityNERAgent()
        self.high = HighComplexityLLMAgent(model=llm_model, device=device)
        self.formatter = FormatterAgent()
        self._strategies = {
            "low": self.low.run,
            "medium": self.medium.run,
            "high": self.high.run,
        }

    def run(self, text: str) -> Dict[str, str]:
        cx = self.complexity.run(text)
        strategy = self.selector.run(cx)
        extractor = self._strategies.get(strategy, self.medium.run)

        names = extractor(text)
        out = self.formatter.run(names)

        return {
            "input": text,
            "complexity": cx.label,
            "confidence": f"{cx.confidence:.2f}",
            "strategy": strategy,
            "output": out,
            "rationale": cx.rationale or "",
        }

# -----------------------------
# Demo & CLI
# -----------------------------
EXAMPLES = [
    ("Jesse Robinson/Greg Phillips/Kishaun Bailey/Kai Asa Savon Wright", "Jesse Robinson/Greg Phillips/Kishaun Bailey/Kai Asa Savon Wright"),
    ("Q luv", "Q luv"),
    ("Pixouu/Abdou Gambetta/Copyright Control", "Pixouu/Abdou Gambetta"),
    ("<Unknown>/Wright, Justyce Kaseem", "Wright/Justyce Kaseem")
    ("UNKNOWN WRITER (999990)", ""),
    ("DJ PALEMBANG/Copyright Control", "DJ PALEMBANG"),
    ("Day & Murray (Bowles, Gaudet, Middleton & Shanahan)", "Day/Murray/Bowles/Gaudet/Middleton/Shanahan"),
    ("King Von,Tee Grizzley,Chopsquad DJ,DJ Bandz", "King Von/Tee Grizzley/Chopsquad DJ/DJ Bandz"),
    ("SUBZEROSWIZ,Yeat", "SUBZEROSWIZ/Yeat"),
]

def run_demo(model: str, device: Optional[str]):
    orch = OrchestratorAgent(llm_model=model, device=device)
    print("\nDemo: Agentic Text Normalization\n" + "-" * 40)
    for raw, expected in EXAMPLES:
        res = orch.run(raw)
        ok = res["output"] == expected
        print(json.dumps({
            "input": raw,
            "expected": expected,
            "got": res["output"],
            "complexity": res["complexity"],
            "strategy": res["strategy"],
            "pass": ok
        }, ensure_ascii=False))

def evaluate_csv(path: str, model: str, device: Optional[str], sample_n: Optional[int] = None):
    import pandas as pd
    df = pd.read_csv(path, keep_default_na=False)
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    orch = OrchestratorAgent(llm_model=model, device=device)

    results = []
    for _, row in df.iterrows():
        raw = str(row.get("raw_comp_writers_text", "") or "")
        expected = str(row.get("CLEAN_TEXT", "") or "")
        res = orch.run(raw)
        got = res["output"]
        results.append({
            "input": raw,
            "expected": expected,
            "got": got,
            "complexity": res["complexity"],
            "strategy": res["strategy"],
            "pass": expected == got
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results_eval.csv", index=False)

    acc = results_df["pass"].mean()
    print(f"\nEvaluation on {len(results_df)} samples")
    print(f"Exact-match Accuracy: {acc:.2%}")
    print("Saved detailed results to results_eval.csv")

def main():
    parser = argparse.ArgumentParser(description="Agentic Text Normalization")
    parser.add_argument("--input", type=str, default=None, help="raw writer string")
    parser.add_argument("--model", type=str, default="google/flan-t5-large", help="HF model id for small LLM")
    parser.add_argument("--device", type=str, default=None, help="device index or 'cpu'")
    parser.add_argument("--demo", action="store_true", help="run built-in examples")
    parser.add_argument("--eval_csv", type=str, help="Path to CSV (columns: raw_comp_writers_text, CLEAN_TEXT)")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for --eval_csv")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.model, args.device)
        return

    if args.eval_csv:
        evaluate_csv(args.eval_csv, args.model, args.device, args.sample)
        return

    if not args.input:
        print("Provide --input, --demo, or --eval_csv")
        return

    orch = OrchestratorAgent(llm_model=args.model, device=args.device)
    res = orch.run(args.input)
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
