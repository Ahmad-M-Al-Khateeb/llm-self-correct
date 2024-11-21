### Skeleton code for SCoRe implementation
import json
import torch.nn.functional as F
from eval_answer import evaluate_answer

# First and second stage prompts directly taken from SCoRe appendix
first_stage_prompt = ( 'You are a math expert. When you respond, respond only with the Solution of the final Problem, ' 
                       'thinking step by step. At the end of the Solution, when you give your final answer, write it in '
                       'the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct."')

second_stage_prompt = ('There might be an error in the solution above because of lack of understanding of the question.' 
                       'Please correct the error, if any, and rewrite the solution. Only output the final solution! At '
                       'the end of the Solution, when you give your final answer, write it in the form "Final Answer: The '
                       'final answer is \\boxed{answer}. I hope it is correct."')

    
def first_stage(model, problem):
    prompt = first_stage_prompt + problem
    return model_predict(model, prompt)

def second_stage(model, problem):
    prompt = second_stage_prompt + problem
    return model_predict(model, prompt)

def solve(model, problem):
    # First stage response
    y1 = first_stage(model, problem)
    
    # Second stage response
    y2 = second_stage(model, problem + y1)

    return y1, y2
    
def read_json(fname):
    with open(fname, 'r') as fp:
        try:
            return json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e

def evaluate_response(ground_truth, response):
    """Check how closely response matches ground truth solution."""
    pass

def kl_divergence(response, base_response):
    """Calculating KL-divergence current response and base model's response (target)"""
    resp_logits = response.logits
    base_logits = base_response.logits

    resp = F.softmax(resp_logits)
    base = F.softmax(base_logits)

    return F.kl_div(resp, base, reduction="batchmean")


def main():
    model = None # pytorch model?
    base_model = None # base (unchanged) model

    # KL-divergance loss hyperparameters
    b1 = 1    # ? need to tune, placeholder for now
    b2= 10*b1 # claimed this produced good results in paper

    # Loading problem form math dataset
    fname = "104.json"
    problem_data = read_json(fname)

    base_model_answer = first_stage(base_model, problem_data["problem"])

    y1, y2 = solve(model, problem_data["problem"])

    kl_loss = kl_divergence(y1, base_model_answer)
    eval = evaluate_answer(y2, problem_data["solution"])

    # Calculate final reward

### Temporary for ollama models
# ollama model code
def model_predict(model, prompt):
    import ollama
    response = ollama.chat(model="llama3.2", messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response["message"]["content"]

def test_ollama(model, question_file):
    problem_data = read_json(question_file)
    y1, y2 = solve(model, problem_data["problem"])
    answers = [y1, y2]

    eval1 = evaluate_answer(y1, problem_data["solution"])
    eval2 = evaluate_answer(y2, problem_data["solution"])
    evals = [eval1, eval2]
    
    return evals, answers


if __name__ == "__main__":
    model = "llama3.2"
    problem_file = "MATH/train/number_theory/8.json"
    evals, answers = test_ollama(model, problem_file)

    print(evals)
    print(answers[0])
    print("Second answer:\n", answers[1])
    







