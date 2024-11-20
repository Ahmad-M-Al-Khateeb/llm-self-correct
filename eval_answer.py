from utils import last_boxed_only_string
from math_equivalence import is_equiv

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    

def evaluate_answer(model_output, problem_solution):
    
    model_answer = remove_boxed(last_boxed_only_string(model_output))
    correct_answer = remove_boxed(last_boxed_only_string(problem_solution))

    return is_equiv(model_answer, correct_answer)

if __name__ == "__main__":
    model_answer = """Solution:
    \[\left(\frac{7}{8}\right)^3 \cdot \left(\frac{7}{8}\right)^{-3} = \left(\frac{7}{8}\right)^{3 + (-3)} = \left(\frac{7}{8}\right)^0 = 1.\] 
    Final Answer: The final answer is \\boxed{1}\. I hope it is correct."""

    problem_solution = "By definition, if $a$ is nonzero, then $a^{-3}$ is the reciprocal of $a^3$.  So, $\\left(\\frac78\\right)^3$ and $\\left(\\frac78\\right)^{-3}$ are reciprocals.  Therefore, their product is $\\boxed{1}$."

    print(evaluate_answer(model_answer, problem_solution))