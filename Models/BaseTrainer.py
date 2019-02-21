# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

class BaseTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = False;
        if self.opt['cuda'] == True:
            self.use_cuda = True
            print('Using Cuda\n') 
        else:
            self.use_cuda = False
            print('Using CPU\n')

        self.is_official = 'OFFICIAL' in self.opt
        # Make sure raw text feature files are ready
        self.use_spacy = 'SPACY_FEATURE' in self.opt
        self.opt['logFile'] = 'log.txt'

        opt['FEATURE_FOLDER'] = 'conf~/' + ('spacy_intermediate_feature~/' if self.use_spacy else 'intermediate_feature~/')
        opt['FEATURE_FOLDER'] = os.path.join(opt['datadir'], opt['FEATURE_FOLDER'])

    def log(self, s):
        # In official case, the program does not output logs
        if self.is_official:
            return

        with open(os.path.join(self.saveFolder, self.opt['logFile']), 'a') as f:
            f.write(s + '\n')
        print(s)

    def getSaveFolder(self):
        runid = 1
        while True:
            saveFolder = os.path.join(self.opt['datadir'], 'conf~', 'run_' + str(runid))
            if not os.path.exists(saveFolder):
                self.saveFolder = saveFolder
                os.makedirs(self.saveFolder)
                print('Saving logs, model and evaluation in ' + self.saveFolder)
                return
            runid = runid + 1    
  
    # save copy of conf file 
    def saveConf(self):
        with open(self.opt['confFile'], encoding='utf-8') as f:
            with open(os.path.join(self.saveFolder, 'conf_copy'), 'w', encoding='utf-8') as fw:
                for line in f:
                    fw.write(line + '\n')

    def train(self): 
        pass
 
    def load(self):
        pass
