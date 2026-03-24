import gymnasium as gym
import re
from datasets import load_dataset
from verl.protocol import DataProto, DataProtoItem
import numpy as np

import copy
import random
        
def transform_dataproto_to_sample(sample):
    """
    Transform a DataProto or DataProtoItem to the format expected by MMLUEnv.
    
    Args:
        sample: Either a DataProto (will extract first item) or DataProtoItem
        
    Returns:
        dict: Transformed sample with keys: 'question', 'subject', 'choices', 'answerKey', 'id'
    """
    # Handle DataProto - extract first item if it's a batch
    if isinstance(sample, DataProto):
        if len(sample) > 0:
            sample = sample[0]
        else:
            raise ValueError("DataProto is empty")
    
    if isinstance(sample, DataProtoItem):
        data = sample.non_tensor_batch
    elif isinstance(sample, dict):
        data = sample
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")
    
    question = data.get('question', '')
    subject = data.get('subject', '')
    choices_raw = data.get('choices', [])
    answer_raw = data.get('answer', None)
    uid = data.get('uid', '')
    
    if isinstance(choices_raw, dict):
        choices = choices_raw
    elif isinstance(choices_raw, list):
        labels = ['A', 'B', 'C', 'D'][:len(choices_raw)]
        choices = {'label': labels, 'text': choices_raw}
    else:
        raise ValueError(f"Unexpected choices format: {type(choices_raw)}")
    
    map_idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    if answer_raw is None:
        answerKey = 'UNKNOWN'
    elif isinstance(answer_raw, int) and answer_raw in map_idx_to_label:
        answerKey = map_idx_to_label[answer_raw]
    elif isinstance(answer_raw, str) and answer_raw.upper() in ['A', 'B', 'C', 'D']:
        answerKey = answer_raw.upper()
    else:
        try:
            idx = int(answer_raw)
            answerKey = map_idx_to_label[idx] if idx in map_idx_to_label else str(answer_raw)
        except (ValueError, TypeError):
            answerKey = str(answer_raw)
    
    sample_id = uid if uid else f"mmlu_sample_{hash(str(data))}"
    
    return {
        'question': question,
        'subject': subject,
        'choices': choices,
        'answerKey': answerKey,
        'id': sample_id
    }


# ---------------------------------------------------------------------------
# Leakage detection & masking
# ---------------------------------------------------------------------------

# Patterns that constitute leakage in the dreaming output
_LEAKAGE_PATTERNS = [
    # Option labels like "A:", "B.", "Option A", "Choice 2", "Statement X"
    re.compile(r'\b(?:Option|Choice|Statement|Answer)\s*[A-D1-4]\b', re.IGNORECASE),
    re.compile(r'\b[A-D]\s*[:\.\)]', re.IGNORECASE),
    # Explicit answer conclusions
    re.compile(r'\b(?:the\s+)?(?:correct\s+)?answer\s+is\b', re.IGNORECASE),
    re.compile(r'\bTherefore[:\s]*[A-D]\b', re.IGNORECASE),
    re.compile(r'\bThus[:\s]*[A-D]\b', re.IGNORECASE),
    re.compile(r'\bHence[:\s]*[A-D]\b', re.IGNORECASE),
    # Bracketed verdicts
    re.compile(r'\[(?:TRUE|FALSE|CORRECT|INCORRECT|YES|NO)\]', re.IGNORECASE),
    re.compile(r'\[(?:[A-D])\]'),
]

# A neutral replacement token that won't disturb tokenizer distributions
_MASK_TOKEN = "[MASK]"


def detect_leakage(text: str) -> tuple[bool, list[str]]:
    """
    Detect leakage patterns in the dreaming output.

    Returns:
        (has_leakage, list_of_matched_strings)
    """
    matches = []
    for pattern in _LEAKAGE_PATTERNS:
        found = pattern.findall(text)
        matches.extend(found)
    return (len(matches) > 0, matches)


def mask_leakage(text: str) -> tuple[str, bool, list[str]]:
    """
    Replace leakage patterns with _MASK_TOKEN.

    Returns:
        (masked_text, has_leakage, matched_strings)
    """
    has_leakage, matches = detect_leakage(text)
    masked = text
    for pattern in _LEAKAGE_PATTERNS:
        masked = pattern.sub(_MASK_TOKEN, masked)
    return masked, has_leakage, matches


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MMLUEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_internal_idx = 0
        self.current_sample = None
        self.phase = None
        self.z_content = ""
        
    def reset(self, sample, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_sample = transform_dataproto_to_sample(sample)
        self.phase = 'dreaming'
        self.z_content = ""
        
        question = self.current_sample['question']
        subject = self.current_sample['subject'].replace("_", " ").title()
        
        choices_raw = self.current_sample['choices']
        choices_raw_copy = copy.deepcopy(choices_raw)
        random.shuffle(choices_raw_copy['text'])
        choices_text = "\n".join([f"{l}: {t}" for l, t in zip(
            choices_raw_copy['label'],
            choices_raw_copy['text']
        )])

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a creative Knowledge Synthesizer specializing in {{subject}}. "
                    "Your goal is to write a 'Context Description' that allows a downstream Solver "
                    "to deduce the correct answer *without* seeing the original question.\n\n"
                    
                    "**CORE INSTRUCTIONS:**\n"
                    "1. **Zero Data Leakage**: Completely strip away all original nouns, names, and "
                    "specific phrasing. Translate the core logic into an entirely different domain "
                    "(e.g., use fluid dynamics to explain economics, or clockwork mechanisms to "
                    "explain biology).\n"
                    "2. **Metaphorical Precision**: Your analogy must mathematically or logically "
                    "align with the correct answer perfectly, while making the incorrect options "
                    "impossible within the metaphor's rules.\n"
                    "3. **Strict Format Constraints** — violations will be penalised:\n"
                    "   - DO NOT repeat the question text.\n"
                    "   - DO NOT mention the answer letter or any explicit option text.\n"
                    "   - DO NOT use option labels such as 'A:', 'B.', 'Option 1', 'Choice C', "
                    "or 'Statement X'.\n"
                    "   - DO NOT write explicit conclusions such as 'The answer is…', "
                    "'Therefore: C', 'Hence the correct choice is…'.\n"
                    "   - DO NOT use bracketed verdicts such as '[TRUE]', '[FALSE]', or '[C]'."
                ).format(subject=subject)
            },
            {
                "role": "user",
                "content": (
                    f"Domain: {subject}\n"
                    f"Question: {question}\n"
                    f"Options:\n{choices_text}\n\n"
                    "Task: Output ONLY the Context Description that satisfies the above instructions."
                )
            }
        ]
        
        return messages, {"temperature": 1.2, "max_tokens": 2048, "phase": "dreaming"}

    def step(self, action_text):
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        def _parse_output(text):
            end_tag = "</think>"
            end_idx = text.find(end_tag)
            if end_idx != -1:
                thought_content = text[:end_idx].replace("<think>", "").strip()
                z_content = text[end_idx + len(end_tag):].lstrip('\r\n')
                if not z_content:
                    z_content = "No context generated."
            else:
                thought_content = ""
                z_content = text.lstrip('\r\n') 
            return thought_content, z_content

        if self.phase == 'dreaming':
            raw_output = action_text
            raw_z = raw_output.lstrip('\r\n') or "No context generated."

            # Leakage detection & masking
            masked_z, has_leakage, leakage_matches = mask_leakage(raw_z)
            self.z_content = masked_z
            
            self.phase = 'answering'
            
            choices = self.current_sample['choices']
            choice_str = "\n".join([f"{l}: {t}" for l, t in zip(choices['label'], choices['text'])])
            subject = self.current_sample['subject'].replace("_", " ").title()

            messages = [
                {
                    "role": "system", 
                    "content": (
                        f"You are an expert in {{subject}}. You will be given a 'Context Description' "
                        "(a scenario, derivation, or set of facts) and a list of Options.\n"
                        "INSTRUCTIONS:\n"
                        "1. Read the Context Description carefully. It describes the solution path "
                        "or the unique properties of the correct answer.\n"
                        "2. Select the option that best matches the description or completes the "
                        "logic derived from the context.\n"
                        "3. Analyze inside <think>...</think> tags, then output ONLY the single "
                        "capital letter label (A, B, C, or D)."
                    ).format(subject=subject)
                },
                {
                    "role": "user", 
                    "content": (
                        f"Context Description:\n{self.z_content} \n\n"
                        f"Options:\n{choice_str}\n\n"
                        "Output Format: <think> reasoning </think> Label"
                    )
                }
            ]
            
            info = {
                "raw_dream_output": raw_output,
                "answering_messages": messages,
                "z_content": self.z_content,
                "temperature": 0.1,
                "max_tokens": 2048,
                "phase": "answering",
                "metrics": {
                    "dreaming/has_leakage": int(has_leakage),
                    "dreaming/leakage_count": len(leakage_matches),
                },
            }
            
            return messages, reward, done, truncated, info
            
        elif self.phase == 'answering':
            answer_thought, clean_answer_text = _parse_output(action_text)
            
            pred_label = self._extract_answer(clean_answer_text)
            true_label = self.current_sample['answerKey']
            
            reward = 1.0 if pred_label == true_label else 0.0
            done = True
            
            info = {
                'correct_answer': true_label,
                'model_answer': pred_label,
                'dream_z': self.z_content,
                'answer_thought_trace': answer_thought,
                'question': self.current_sample['question'],
                'id': self.current_sample['id'],
                'temperature': 0.1,
                'max_tokens': 2048,
                'phase': "answering",
                "metrics": {},
            }
            
            return None, reward, done, truncated, info

    def _extract_answer(self, text):
        match = re.search(r'\b([A-D])\b', text.upper())
        if match:
            return match.group(1)
        return "UNKNOWN"
    
def get_mmlu_env(config):
    return MMLUEnv(config)