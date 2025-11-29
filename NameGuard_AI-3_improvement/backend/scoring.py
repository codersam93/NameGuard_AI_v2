from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional, NamedTuple

import pandas as pd
from indian_namematch import fuzzymatch

try:
    from backend.models import NameEvaluationInput, NameEvaluationResult, DecisionLabel, RuleFlag
except ImportError:
    from models import NameEvaluationInput, NameEvaluationResult, DecisionLabel, RuleFlag

# ---------------------------
# Paths
# ---------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------
# Text helpers & basic normalization
# ---------------------------

SUFFIXES = [
    "private limited",
    "pvt ltd",
    "limited",
    "ltd",
    "opc private limited",
    "llp",
    "producer company limited",
]


def normalize_name(raw: str) -> str:
    """Normalize company name for easier token comparison.

    - Lowercase
    - Strip common MCA entity suffixes
    - Collapse multiple spaces
    """

    base = raw.strip().lower()
    for suf in SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)].strip()
            break
    return " ".join(base.split())


def simple_phonetic_key(raw: str) -> str:
    s = normalize_name(raw)
    repl = [
        ("ph", "f"),
        ("bh", "b"),
        ("kh", "k"),
        ("gh", "g"),
        ("ch", "c"),
        ("sh", "s"),
        ("ss", "s"),
        ("aa", "a"),
        ("ee", "i"),
        ("oo", "u"),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    return "".join(c for c in s if c.isalnum())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b)) or 1
    return 1.0 - dist / max_len


# ---------------------------
# indian-namematch helpers (phonetic matching)
# ---------------------------


def phonetic_match(a: str, b: str) -> bool:
    """Return True if `a` and `b` are considered a phonetic match.

    Uses indian-namematch.fuzzymatch.single_compare under the hood.
    """
    try:
        result = fuzzymatch.single_compare(str(a), str(b))
        return isinstance(result, str) and result.lower().startswith("match")
    except Exception:
        return False


# ---------------------------
# Data Structures for Optimization
# ---------------------------

class IndexedTerm(NamedTuple):
    term: str
    key: str
    metadata: Dict  # Store severity, language, etc.

class IndexedData:
    def __init__(self):
        self.offensive_words: List[IndexedTerm] = []
        self.prohibited_terms: List[IndexedTerm] = []
        self.existing_companies: List[IndexedTerm] = []
        self.historical_decisions: List[IndexedTerm] = []
        self.existing_company_keys: Set[str] = set()

# Global data instance
DATA = IndexedData()

# ---------------------------
# Data loading helpers
# ---------------------------

def _load_data():
    global DATA
    
    # 1. Offensive Words
    path = DATA_DIR / "offensive_words_multilang.xlsx"
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                w = str(row.get("word", "")).strip().lower()
                if not w: continue
                lang = str(row.get("language", "")).strip().lower() or "unknown"
                try: severity = float(row.get("severity", 1))
                except: severity = 1.0
                severity = max(0.0, min(1.0, severity))
                
                DATA.offensive_words.append(IndexedTerm(
                    term=w,
                    key=simple_phonetic_key(w),
                    metadata={"lang": lang, "severity": severity}
                ))
        except Exception: pass

    if not any(t.term == "chutiya" for t in DATA.offensive_words):
        DATA.offensive_words.append(IndexedTerm("chutiya", simple_phonetic_key("chutiya"), {"lang": "hi", "severity": 1.0}))

    # 2. Prohibited Words
    path = DATA_DIR / "prohibited_words.xlsx"
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                term = str(row.get("word", "")).strip().lower()
                if not term: continue
                try: severity = float(row.get("severity", 1))
                except: severity = 1.0
                severity = max(0.0, min(1.0, severity))
                
                DATA.prohibited_terms.append(IndexedTerm(
                    term=term,
                    key=simple_phonetic_key(term),
                    metadata={"severity": severity}
                ))
        except Exception: pass

    if not DATA.prohibited_terms:
        defaults = ["government of india", "central government", "reserve bank of india", "bharat", "india", "national", "republic"]
        for term in defaults:
            DATA.prohibited_terms.append(IndexedTerm(term, simple_phonetic_key(term), {"severity": 1.0}))

    # 3. Existing Companies
    path = DATA_DIR / "existing_companies_sample.xlsx"
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                name = str(row.get("name") or "").strip()
                if not name: continue
                
                key = str(row.get("phonetic_key", "")).strip().lower()
                if not key:
                    base_name = str(row.get("normalized_name") or name).strip()
                    key = simple_phonetic_key(base_name)
                
                DATA.existing_companies.append(IndexedTerm(name, key, {}))
                DATA.existing_company_keys.add(key)
        except Exception: pass

    fallback_names = [
        "Tata Consultancy Services", "Infosys Technologies", "Bharat Heavy Electricals",
        "Flipkart", "Flipkart Internet Private Limited", "Accenture", "Accenture Solutions Private Limited"
    ]
    for n in fallback_names:
        key = simple_phonetic_key(n)
        DATA.existing_companies.append(IndexedTerm(n, key, {}))
        DATA.existing_company_keys.add(key)

    # 4. Historical Decisions
    path = DATA_DIR / "historical_name_decisions.xlsx"
    if path.exists():
        try:
            df = pd.read_csv(path)
            df["decision_dt"] = pd.to_datetime(df.get("decision_timestamp"), utc=True, errors="coerce")
            
            for _, row in df.iterrows():
                candidate = str(row.get("name_normalized") or row.get("name_original") or "").strip()
                if not candidate: continue
                
                key = simple_phonetic_key(candidate)
                decision = str(row.get("decision") or "").strip().lower()
                
                DATA.historical_decisions.append(IndexedTerm(
                    term=candidate,
                    key=key,
                    metadata={
                        "decision_dt": row.get("decision_dt"),
                        "decision": decision,
                        "score_at_time": row.get("score_at_time")
                    }
                ))
                
                # If a name was accepted in history, treat it as an EXISTING COMPANY
                if decision == "accepted":
                    DATA.existing_companies.append(IndexedTerm(candidate, key, {}))
                    DATA.existing_company_keys.add(key)
                    
        except Exception: pass

# Load data on module import
_load_data()


# Leader / political-related patterns
LEADER_INITIALS: Set[str] = {"nm", "ndm", "n.d.m", "n.d.m.", "n.m.", "n m", "n d m"}
LEADER_PHONETIC_KEYS: Set[str] = {
    simple_phonetic_key("Modi"),
    simple_phonetic_key("NaMo"),
    simple_phonetic_key("Narendra Modi"),
    simple_phonetic_key("Narendra Damodardas Modi"),
}


# ---------------------------
# Optimized Matching Helpers
# ---------------------------

def check_phonetic_match_optimized(name: str, candidates: List[IndexedTerm], threshold: float = 0.4) -> List[IndexedTerm]:
    """
    Find matches in candidates list.
    First filters by key similarity, then runs expensive phonetic_match.
    """
    name_key = simple_phonetic_key(name)
    matches = []
    
    for item in candidates:
        # Fast filter: key similarity
        if similarity(name_key, item.key) < threshold:
            continue
            
        if phonetic_match(name, item.term):
            matches.append(item)
            
    return matches

def check_phonetic_match_any_token_optimized(name: str, candidates: List[IndexedTerm], threshold: float = 0.4) -> List[IndexedTerm]:
    """
    Check if ANY token in name matches any candidate.
    """
    tokens = normalize_name(name).split()
    matches = []
    
    # Pre-compute token keys
    token_keys = [(t, simple_phonetic_key(t)) for t in tokens]
    
    for item in candidates:
        matched = False
        for token, t_key in token_keys:
            if similarity(t_key, item.key) < threshold:
                continue
            
            if phonetic_match(token, item.term):
                matches.append(item)
                matched = True
                break 
        
    return matches


# ---------------------------
# Historical acceptance (frequency + recency weighted)
# ---------------------------


def historical_acceptance_from_phonetics(name: str) -> Optional[float]:
    """
    Calculates the likelihood of acceptance based on historical decisions.
    
    Logic:
    - Finds all phonetic matches in historical data.
    - Calculates a weighted average of outcomes (Accepted=1.0, Rejected=0.0).
    - Weights are determined by Recency.
    - Frequency is inherently handled by summing up all matching occurrences.
      (e.g., 5 rejections contribute 5x the weight of 1 rejection, adjusted for recency).
    
    Returns:
        Float between 0.0 and 1.0 representing historical approval rate.
        None if no history found.
    """
    if not DATA.historical_decisions:
        return None

    # Use optimized matching
    matches = check_phonetic_match_optimized(name, DATA.historical_decisions, threshold=0.4)
    
    if not matches:
        return None

    now = datetime.now(timezone.utc)
    numer = 0.0
    denom = 0.0

    for item in matches:
        dt_val = item.metadata.get("decision_dt")
        if dt_val is None or pd.isna(dt_val):
            continue

        # Recency Weight Calculation
        # More recent = Higher weight
        age_days = max(0.0, (now - dt_val).total_seconds() / 86400.0)
        age_years = age_days / 365.25
        
        # Decay function: 1 / (1 + age_years)
        # 0 years -> 1.0
        # 1 year -> 0.5
        # 2 years -> 0.33
        w_recency = 1.0 / (1.0 + age_years)

        decision_str = item.metadata.get("decision")
        score_at_time = item.metadata.get("score_at_time")
        
        if isinstance(score_at_time, (int, float)):
            outcome = float(score_at_time)
        else:
            outcome = 1.0 if decision_str == "accepted" else 0.0

        # Accumulate weighted outcome
        # Frequency is handled here: each occurrence adds to the sum
        numer += w_recency * outcome
        denom += w_recency

    if denom == 0.0:
        return None

    return numer / denom


# ---------------------------
# Scoring engine
# ---------------------------


def compute_hard_rule_flags(name: str) -> List[RuleFlag]:
    flags: List[RuleFlag] = []
    lowered = name.lower()

    # 1) Prohibited / reserved phrases
    for item in DATA.prohibited_terms:
        if item.term in lowered:
            flags.append(RuleFlag(
                code="prohibited_or_reserved_term",
                description=f"Contains term or phrase '{item.term}' that is closely associated with Government.",
                severity=item.metadata["severity"]
            ))

    # 2) Offensive / abusive words
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    offensive_dict = {item.term: item for item in DATA.offensive_words}
    
    for token in tokens:
        if token in offensive_dict:
            item = offensive_dict[token]
            flags.append(RuleFlag(
                code="obscene_or_offensive",
                description=f"Contains offensive or abusive word '{token}' in language '{item.metadata['lang']}'.",
                severity=item.metadata["severity"]
            ))

    # 3) Phonetic match against offensive words
    has_offensive_flag = any(f.code == "obscene_or_offensive" for f in flags)
    if not has_offensive_flag:
        matches = check_phonetic_match_any_token_optimized(name, DATA.offensive_words)
        for item in matches:
            flags.append(RuleFlag(
                code="obscene_or_offensive",
                description=f"Contains word that is phonetically similar to offensive term '{item.term}' in language '{item.metadata['lang']}'.",
                severity=max(item.metadata["severity"], 0.9)
            ))
            break 

    # 4) Phonetic match against prohibited terms
    has_prohibited_flag = any(f.code == "prohibited_or_reserved_term" for f in flags)
    if not has_prohibited_flag:
        matches = check_phonetic_match_any_token_optimized(name, DATA.prohibited_terms)
        for item in matches:
            flags.append(RuleFlag(
                code="prohibited_or_reserved_term",
                description=f"Contains expression that is phonetically close to reserved term '{item.term}'.",
                severity=max(item.metadata["severity"], 0.9)
            ))
            break

    # 5) Near-identical to existing companies
    matches = check_phonetic_match_optimized(name, DATA.existing_companies)
    for item in matches:
        flags.append(RuleFlag(
            code="too_similar_existing_company",
            description="Very high phonetic similarity to an existing registered company name.",
            severity=0.9
        ))
        break

    # 6) Political leader initials
    for token in tokens:
        compact = token.replace(".", "").lower()
        if compact in {i.replace(" ", "") for i in LEADER_INITIALS}:
            flags.append(RuleFlag(
                code="political_or_leader_reference_initials",
                description="Appears to use initials or monogram referring to a political leader.",
                severity=0.9
            ))
            break

    # 7) Phonetic similarity to PM name
    for token in tokens:
        if token.lower() in {"name", "company", "enterprises", "solutions", "services"}:
            continue
        token_key = simple_phonetic_key(token)
        for leader_key in LEADER_PHONETIC_KEYS:
            sim = similarity(token_key, leader_key)
            if sim >= 0.75:
                flags.append(RuleFlag(
                    code="political_or_leader_reference_phonetic",
                    description="Contains a word that is phonetically very similar to the name of the Prime Minister.",
                    severity=0.85
                ))
                break
        else:
            continue
        break

    return flags


def uniqueness_score(name: str) -> float:
    key = simple_phonetic_key(name)
    if not DATA.existing_company_keys:
        return 1.0
    
    sims = [similarity(key, other) for other in DATA.existing_company_keys]
    max_sim = max(sims) if sims else 0.0
    
    if max_sim < 0.3:
        penalty = 0.0
    elif max_sim > 0.8:
        penalty = 1.0
    else:
        penalty = (max_sim - 0.3) / (0.8 - 0.3)
        
    return 1.0 - penalty


def aggregate_score(name: str) -> Tuple[float, List[RuleFlag], List[str]]:
    flags = compute_hard_rule_flags(name)
    explanations: List[str] = []

    # Base score starts high for a clean name
    score = 0.95

    # 1. Uniqueness Penalty
    uniq = uniqueness_score(name)
    score = score * uniq
    explanations.append(f"Distinguishability from existing names is estimated at {uniq * 100:.0f}%.")

    # 2. Historical Rejection Penalty
    hist_prob = historical_acceptance_from_phonetics(name)
    if hist_prob is not None:
        # If historical approval rate is low (< 50%), apply penalty
        if hist_prob < 0.5:
            score = min(score, 0.4)
            explanations.append(f"Historical data indicates similar names have been rejected (Approval rate: {hist_prob * 100:.0f}%).")
        
    hard_severity = sum(f.severity for f in flags)

    has_high_offensive = any(f.code == "obscene_or_offensive" and f.severity >= 0.9 for f in flags)
    has_reserved = any(f.code == "prohibited_or_reserved_term" for f in flags)
    has_too_similar = any(f.code == "too_similar_existing_company" for f in flags)
    has_political_initials = any(f.code == "political_or_leader_reference_initials" for f in flags)
    has_political_phonetic = any(f.code == "political_or_leader_reference_phonetic" for f in flags)

    if hard_severity:
        base_penalty = min(0.7, 0.2 * hard_severity)
        score = max(0.0, score * (1.0 - base_penalty))
        explanations.append("MCA rule-based checks reduce the estimated approval probability.")

    if has_high_offensive:
        score = min(score, 0.05)
        explanations.append("Contains highly offensive language.")

    if has_reserved:
        score = min(score, 0.25)
        explanations.append("Includes Government-reserved expressions.")

    if has_too_similar:
        score = min(score, 0.2)
        explanations.append("Too similar to existing registered company name.")

    if has_political_initials or has_political_phonetic:
        score = min(score, 0.2)
        explanations.append("References a political leader.")

    score = min(1.0, max(0.0, score))

    return score, flags, explanations


def label_from_score(prob: float) -> DecisionLabel:
    if prob >= 75.0:
        return DecisionLabel.HIGH
    if prob >= 40.0:
        return DecisionLabel.MEDIUM
    return DecisionLabel.LOW


def evaluate_single(input_item: NameEvaluationInput) -> NameEvaluationResult:
    score_0_1, flags, explanations = aggregate_score(input_item.name)
    probability = round(score_0_1 * 100, 2)
    label = label_from_score(probability)
    return NameEvaluationResult(
        name=input_item.name,
        priority=input_item.priority,
        acceptance_probability=probability,
        decision_label=label,
        rule_flags=flags,
        explanations=explanations,
    )


def evaluate_names(inputs: Iterable[NameEvaluationInput]) -> List[NameEvaluationResult]:
    return [evaluate_single(i) for i in inputs]
