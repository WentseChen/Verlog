import gymnasium as gym
import re
from datasets import load_dataset
from verl.protocol import DataProto, DataProtoItem
import numpy as np

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
            sample = sample[0]  # Extract first item as DataProtoItem
        else:
            raise ValueError("DataProto is empty")
    
    # Extract data from non_tensor_batch
    if isinstance(sample, DataProtoItem):
        data = sample.non_tensor_batch
    elif isinstance(sample, dict):
        # Already a dict, use directly
        data = sample
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")
    
    # Extract required fields
    question = data.get('question', '')
    subject = data.get('subject', '')
    choices_raw = data.get('choices', [])
    answer_raw = data.get('answer', None)
    uid = data.get('uid', '')
    
    # Transform choices to expected format
    # If choices is already a dict with 'label' and 'text', use it
    # Otherwise, assume it's a list of strings and create labels
    if isinstance(choices_raw, dict):
        choices = choices_raw
    elif isinstance(choices_raw, list):
        # Create labels A, B, C, D for the choices
        labels = ['A', 'B', 'C', 'D'][:len(choices_raw)]
        choices = {
            'label': labels,
            'text': choices_raw
        }
    else:
        raise ValueError(f"Unexpected choices format: {type(choices_raw)}")
    
    # Transform answer to answerKey
    # If answer is an integer (0-3), convert to A-D
    # If answer is already A-D, use it directly
    map_idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    if answer_raw is None:
        answerKey = 'UNKNOWN'
    elif isinstance(answer_raw, int) and answer_raw in map_idx_to_label:
        answerKey = map_idx_to_label[answer_raw]
    elif isinstance(answer_raw, str) and answer_raw.upper() in ['A', 'B', 'C', 'D']:
        answerKey = answer_raw.upper()
    else:
        # Try to convert if it's a string representation of int
        try:
            idx = int(answer_raw)
            if idx in map_idx_to_label:
                answerKey = map_idx_to_label[idx]
            else:
                answerKey = str(answer_raw)
        except (ValueError, TypeError):
            answerKey = str(answer_raw)
    
    # Use uid as id, or generate one if not available
    sample_id = uid if uid else f"mmlu_sample_{hash(str(data))}"
    
    return {
        'question': question,
        'subject': subject,
        'choices': choices,
        'answerKey': answerKey,
        'id': sample_id
    }


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
        
        # Transform DataProto/DataProtoItem to expected format
        self.current_sample = transform_dataproto_to_sample(sample)
        self.phase = 'dreaming'
        self.z_content = ""
        
        question = self.current_sample['question']
        subject = self.current_sample['subject'].replace("_", " ").title() # e.g. "College Mathematics"
        
        choices_text = "\n".join([f"{l}: {t}" for l, t in zip(
            self.current_sample['choices']['label'],
            self.current_sample['choices']['text']
        )])

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a creative Knowledge Synthesizer specializing in {{subject}}. Your goal is to write a 'Context Description' that allows a downstream Solver to deduce the correct answer *without* seeing the original question.\n\n"
                    
                    "**CORE INSTRUCTIONS:**\n"
                    "1. **Associative & Creative Reasoning**: Do not just list facts. Use lateral thinking to connect {{subject}} concepts with universal mental models or analogies from other fields to reinforce the core logic.\n"
                    "2. **Implicit Guidance**: Describe the precise conditions, formulas, or historical context that validate the correct option and invalidate others.\n"
                    "3. **Strict Constraints**: \n"
                    "   - DO NOT repeat the question text.\n"
                    "   - DO NOT mention the answer letter or explicit option text.\n"
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
        
        # --- Helper: Extract Thought vs Content ---
        def _parse_output(text):
            end_tag = "</think>"
            
            end_idx = text.find(end_tag)
            
            if end_idx != -1:
                thought_part = text[:end_idx]
                thought_content = thought_part.replace("<think>", "").strip()
                
                raw_content = text[end_idx + len(end_tag):]
                
                z_content = raw_content.lstrip('\r\n')
                
                if not z_content:
                     z_content = "No context generated."
            else:
                thought_content = ""
                z_content = text.lstrip('\r\n') 
                
            return thought_content, z_content

        if self.phase == 'dreaming':
            raw_output = action_text
            # z_content is now the entire response (no parsing needed)
            self.z_content = raw_output
            
            if not self.z_content:
                self.z_content = "No context generated."

            self.phase = 'answering'
            
            choices = self.current_sample['choices']
            choice_str = "\n".join([f"{l}: {t}" for l, t in zip(choices['label'], choices['text'])])
            subject = self.current_sample['subject'].replace("_", " ").title()

            # --- MODIFIED: Solver Prompt for MMLU ---
            messages = [
                {
                    "role": "system", 
                    "content": (
                        f"You are an expert in {{subject}}. You will be given a 'Context Description' (a scenario, derivation, or set of facts) and a list of Options.\n"
                        "INSTRUCTIONS:\n"
                        "1. Read the Context Description carefully. It describes the solution path or the unique properties of the correct answer.\n"
                        "2. Select the option that best matches the description or completes the logic derived from the context.\n"
                        "3. Analyze inside <think>...</think> tags, then output ONLY the single capital letter label (A, B, C, or D)."
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
                # Store answering phase messages and z_content as strings
                # They will be tokenized in tool_agent.py
                "answering_messages": messages,
                "z_content": self.z_content,
                "temperature": 0.1,
                "max_tokens": 2048,
                "phase": "answering",
                "metrics": {},
            }
            
            # # Debug prints
            # print(f"Subject: {subject}")
            # print(f"Q: {self.current_sample['question']}")
            # print(f"Dream: {self.z_content}")
            # print("-"*20)
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
            
            # print("Answering Phase:")
            # print("Answer Thought:", answer_thought)
            # print(f"Prediction: {pred_label} | Truth: {true_label} | Reward: {reward}")
            
            return None, reward, done, truncated, info

    def _extract_answer(self, text):
        # MMLU 只有 A, B, C, D
        match = re.search(r'\b([A-D])\b', text.upper())
        if match:
            return match.group(1)
        return "UNKNOWN"
    
def get_mmlu_env(config):
    return MMLUEnv(config)