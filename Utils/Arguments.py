# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

class Arguments:
    def __init__(self, confFile):
        if not os.path.exists(confFile):
            raise Exception("The argument file does not exist: " + confFile)
        self.confFile = confFile

    def is_int(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_bool(self, s):
        return s.lower() == 'true' or s.lower() == 'false'

    def readHyperDriveArguments(self, arguments):
        hyperdrive_opts = {}
        for i in range(0, len(arguments), 2):
            hp_name, hp_value = arguments[i:i+2]
            hp_name = hp_name.replace("--", "")
            if self.is_int(hp_value):
                hp_value = int(hp_value)
            elif self.is_float(hp_value):
                hp_value = float(hp_value)
            hyperdrive_opts[hp_name] = hp_value
        return hyperdrive_opts

    def readArguments(self):
        opt = {}
        with open(self.confFile, encoding='utf-8') as f:
            for line in f:
                l = line.replace('\t', ' ').strip()
                if l.startswith("#"):
                    continue
                parts = l.split()
                if len(parts) == 1:
                    key = parts[0]
                    if not key in opt:
                        opt[key] = True
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1]
                    if not key in opt:
                        opt[key] = value
                        if self.is_int(value):
                            opt[key] = int(value)
                        elif self.is_float(value):
                            opt[key] = float(value)
                        elif self.is_bool(value):
                            opt[key] = value.lower() == 'true'
                    else:
                        print('Warning: key %s already exists' % key)
        return opt
