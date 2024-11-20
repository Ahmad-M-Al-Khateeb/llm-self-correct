import json

def read_json(fname):
    with open(fname, 'r') as file:
        data = json.load(file)

    return data["problem"], data["solution"]

p, s = read_json("104.json")

print(p)
print(s)