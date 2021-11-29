import yaml
import argparse

"""
How to use:

python3 env2req.py --env <env_file> --output <output_file>

Example:
    Linux:
        python3 env2req.py --env ../neuralinterface.yml --output requirements.txt
    Windows:
        python env2req.py --env ../neuralinterface.yml --output requirements.txt

"""

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="File yaml to convert", type=str)
parser.add_argument("--output", help="Output file", type=str)
args = parser.parse_args()

data = None
with open(args.env) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

req = ""
for dependencie in data["dependencies"]:
    if type(dependencie) == type(dict()):
        req +="\n".join(dependencie["pip"])
    else:
        dep = dependencie.split("=")
        req += "{}=={}\n".format(dep[0], dep[1]) 
print(req)
open(args.output,"w").write(req)
print("Successfully requirements created: ", args.output)

