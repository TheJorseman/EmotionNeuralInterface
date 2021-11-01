import yaml
data = None
with open("neuralinterface.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

req = ""
for dependencie in data["dependencies"]:
    if type(dependencie) == type(dict()):
        req +="\n".join(dependencie["pip"])
    else:
        dep = dependencie.split("=")
        req += "{}=={}\n".format(dep[0], dep[1]) 
print(req)
open("requirements.txt","w").write(req)
