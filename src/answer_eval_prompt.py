EVAL_PROMPT = """
You are an evaluator.

Question: {query}
Ground Truth Answer: {ground_truth}
Model Answer: {model_answer}

Task:
1. Identify the MAIN IDEA of the ground truth answer.
2. Check if the model answer expresses the SAME IDEA.

Evaluation Rules:
- Wording does NOT need to match
- Extra details in model answer are OK
- Missing minor details are OK
- Focus only on whether the core concept is correct

Scoring:
- If the main idea matches → Score = 1
- If the meaning is incorrect or missing → Score = 0

Important:
- Be tolerant and semantic
- Do NOT penalize correct answers for phrasing differences

Respond ONLY in this format:
Score: 0 or 1
Reason: <short explanation>
"""