# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def compute_score(solution_str, ground_truth) -> float:
    """Compute score for MMLU multiple-choice questions.
    
    MMLU questions have A, B, C, D as answer choices.
    This function extracts the answer choice from the solution string
    and compares it to the ground truth.
    
    Args:
        solution_str: The solution text containing the answer
        ground_truth: The correct answer (A, B, C, or D)
    
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    # Extract answer using word boundary to match A, B, C, or D
    # This pattern matches single letters A-D as standalone words
    match = re.search(r'\b([A-D])\b', solution_str.upper())
    extracted_answer = match.group(1) if match else None
    
    # Normalize ground truth to uppercase for comparison
    ground_truth_upper = str(ground_truth).upper().strip()
    
    # Return 1.0 if the extracted answer matches ground truth, 0.0 otherwise
    if extracted_answer and extracted_answer == ground_truth_upper:
        return 1.0
    else:
        return 0.0
