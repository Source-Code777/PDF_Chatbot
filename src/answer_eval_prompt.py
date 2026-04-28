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
- If the model answer is "I don't know based on the provided document" → Score = 0
- If the answer is vague, incomplete, or does not answer the question → Score = 0
- The model answer must clearly express the main idea to get Score = 1



Scoring:
- If the main idea matches → Score = 1
- If the meaning is incorrect or missing → Score = 0

Important:
- Do NOT use outside knowledge

Respond ONLY in this format:
Score: 0 or 1
Reason: <short explanation>
"""