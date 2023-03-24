import json


class jsonToClass(object):
    def __init__(self, json_file_path: str = "config.json"):
        with open(json_file_path) as json_file:
            args = json.load(json_file)
        for key in args:
            setattr(self, key, args[key])
            print(f"{key} set to {args[key]}")


args = jsonToClass()
