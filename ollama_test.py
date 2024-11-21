import ollama

question = ('You are a math expert. When you respond, respond only with the Solution of the final Problem, ' 
            'thinking step by step. At the end of the Solution, when you give your final answer, write it in '
            'the form "Final Answer: The final answer is \\boxed{answer}. I hope it is correct."'
            '\n###\n'
            'When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?'
            )

print(question+"\n")

response = ollama.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': question,
  },
])
print(response["message"]["content"])