import re
import os
########
# Read/Write config file
# read_config: Reading ocnfigurations by file name
# -Rulse:
#    [{section name}]: assign sections,section name only contains letters, e.g. [DATA]
#    {keys}={values}: assign key and value to last section specified above, e.g. batch_size=32
#                     can be stand-alone when no sections assigned
#    #discriptions: adding "#" in the beginning will let the discription be neglected,
#                   this can be used in adding annotations, e.g. #this is a note
#    output: an object with configuration variables in sections(or configuration alone)
# -Example:
#    cfg=read_config("config.cfg")
#
# Cfg: Object for configures, keys can be called by "Cfg.keyname"
#    .__init__: input a dictionary and convert to object, e.g. "cfg=Cfg(**config_dictionary)"
#    .__repr__: can be called by the name of object, e.g. "print(cfg)"
#    .write({filename}): write configuations to a file which can be read in the future
########
class Cfg:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __repr__(self):
        print("---Configuration---")
        display(self.__dict__)
        return "------"
    def write(self,filename):
        lines=""
        for keys in list(self.__dict__.keys()):
            if type(self.__dict__[keys])==dict:
                lines+=f"{os.linesep}[{keys}]{os.linesep}"
                dict2=self.__dict__[keys]
                for keys2 in list(dict2.keys()):
                    if type(dict2[keys2])==str:
                        lines+=f"{keys2}=\'{dict2[keys2]}\'{os.linesep}"
                    else:
                        lines+=f"{keys2}={dict2[keys2]}{os.linesep}"
                continue
            if type(self.__dict__[keys])==str:
                lines+=f"{keys}=\'{self.__dict__[keys]}\'{os.linesep}"
            else:
                lines+=f"{keys}={self.__dict__[keys]}{os.linesep}"
        with open(filename,"w") as f:
            f.write(lines)
            
def read_config(filename):
    cfg=dict()
    with open(filename,"r") as f:
        current_section=False
        for line in f.read().splitlines():
            if not line or line[0]=="#":
                continue
            if line[0]=="[":
                section=re.search("\[\S+\]",line)
                cfg[section[0][1:-1]]=dict()
                current_section=section[0][1:-1]
                continue
            if current_section:
                key,val=re.split("=",line)
                cfg[current_section][key]=eval(val)
            else:
                key,val=re.split("=",line) 
                cfg[key]=eval(val)
    return Cfg(**cfg)