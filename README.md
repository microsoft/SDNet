# SDNet

This is the official code for the Microsoft's submission of SDNet model to [CoQA](https://stanfordnlp.github.io/coqa/) leaderboard. It is implemented under PyTorch framework. The related paper to cite is: 

**SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering**, by Chenguang Zhu, Michael Zeng and Xuedong Huang, at https://arxiv.org/abs/1812.03593.

For usage of this code, please follow [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct).

# Directory structure:
* main.py: the starter code

* Models/
  * BaseTrainer.py: Base class for trainer
  * SDNetTrainer.py: Trainer for SDNet, including training and predicting procedures
  * SDNet.py: The SDNet network structure
  * Layers.py: Related network layer functions
  * Bert/
    * Bert.py: Customized class to compute BERT contextualized embedding     
	   * modeling.py, optimization.py, tokenization.py: From Huggingface's PyTorch implementation of BERT
* Utils/
  * Arguments.py: Process argument configuration file
  * Constants.py: Define constants used
  * CoQAPreprocess.py: preprocess CoQA raw data into intermediate binary/json file, including tokenzation, history preprending
  * CoQAUtils.py, General Utils.py: utility functions used in SDNet
  * Timing.py: Logging time

# How to run
Requirement: PyTorch 0.4.0, spaCy 2.0.
The docker we used is available at dockerhub: https://hub.docker.com/r/zcgzcgzcg/squadv2/tags. Please use v3.0 or v4.0.
1. Create a folder (e.g. **coqa**) to contain data and running logs;
2. Create folder **coqa/data** to store CoQA raw data: **coqa-train-v1.0.json** and **coqa-dev-v1.0.json**;
3. Copy the file **conf** from the repo into folder **coqa**;
4. If you want to use BERT-Large, download their model into **coqa/bert-large-uncased**; if you want to use BERT-base, download their model into **coqa/bert-base-cased**;
    * The models can be downloaded from Huggingface: 
      * 'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
      * 'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz"
    * bert-large-uncased-vocab.txt can be downloaded from Google's BERT repository
5. Create a folder **glove** in the same directory of **coqa** and download GloVe embedding **glove.840B.300d.txt** into the folder.

Your directory should look like this:
* coqa/
  * data/
    * coqa-train-v1.0.json
    * coqa-dev-v1.0.json
  * bert-large-uncased/
    * bert-large-uncased-vocab.txt
    * bert_config.json
    * pytorch_model.bin
  * conf  
* glove/
  * glove.840B.300d.txt

Then, execute `python main.py train path_to_coqa/conf`.

If you run for the first time, CoQAPreprocess.py will automatically create folders **conf~/spacy_intermediate_features~** inside **coqa** to store intermediate tokenization results, which will take a few hours.

Every time you run the code, a new running folder **run_idx** will be created inside **coqa/conf~**, which contains running logs, prediction result on dev set, and best model.

# Contact
If you have any questions, please contact Chenguang Zhu, chezhu@microsoft.com
