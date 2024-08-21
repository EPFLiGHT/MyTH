## MyThisYourThat: Interpretable Identification of Systematic Bias in Federated Learning for Biomedical Images

☕ This repository contains code for the `MyTH` model, a novel extension of `ProtoPNet` ([Chen et al.](https://arxiv.org/abs/1806.10574)) for interpretable and privacy-preserving identification of data bias in federated learning for images. The materials necessary to implement the baseline original `ProtoPNet` were adopted from [this repository](https://github.com/cfchen-duke/ProtoPNet).

`MyTH` will also be soon available within [DISCO](https://discolab.ai/#/) web platform for collaborative model training ([GitHub repository](https://github.com/klavdiiaN/disco/tree/ppnet/discojs/src/models/ppnet)).

Here are the steps to reproduce our approach.

### 🗃️ Data preprocessing
______________________
We used the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset of chest X-ray images that can be downloaded [here](https://www.kaggle.com/datasets/ashery/chexpert). After you download and unzip the `archive.zip` file, please, create a new folder named `CheXpert-v1.0-small` and move the unzipped `train` folder inside. Finally, ensure that the `CheXpert-v1.0-small` folder and `train.csv` file are in the current repository.

In this work, we do *one-vs-rest* classification. For this aim, it is needed to separate images belonging to the class of interest (positive class) from all others (negative class). For cardiomegaly and pleural effusion classification, this can be done by running [`preprocessing_cardio.py`](preprocessing_cardio.py) and [`prerocessing_effusion.py`](preprocessing_effusion.py) scripts, respectively. The scripts will create folders with three subfolders containing images for training and validation as well as `push` subfolder with a subset of training images for prototype visualization.

We experimented with two distinct types of data bias. A synthetic bias for the cardiomegaly classification can be added directly within the training scripts described in the next section. However, to introduce a real-world chest drain bias into the pleural effusion class, a separate dataset should be used. To prepare it, run the [`preprocessing_drains.py`](preprocessing_drains.py) script which will output a corresponding `drains` folder. The labels for the presence of chest drains were adopted from [Jiménez-Sánchez A. et al.](https://arxiv.org/abs/2211.04279).

### 💡 Training
____________________________________
1. Centralized Model (**CM**). To train a baseline `ProtoPNet` in a centralized setting, run the [`run_CheXpert_CM.py`](run_CheXpert_CM.py) script providing the following arguments: 
- `-nc` number of classes which is 2 by default in our *one-vs-rest* approach;
- `-e` number of training epochs, we used around 20 for pleural effusion and 30 for cardiomegaly;
- `-c` class name which is either *cardiomegaly* or *effusion*

Example:
```
python run_CheXpert_CM.py -e 22 -c effusion
```
The script outputs two folders:
- `ppnet_chest` with saved models;
- `prot_chest` with saved prototypes.

You need to save these files in a different directory since running other training scripts rewrites the content of these folders.

2. Local Models (**LM**). To train LM, i.e. `ProtoPNet` on smaller local datasets, run the [`run_CheXpert_LM.py`](run_CheXpert_LM.py) script with arguments:
- `-nc`, `-e`, and `-c` as for CM described above;
- `-d` set this flag if you need to distribute the data among the clients;
- `-ncl` number of clients which is 4 by default in our implementation;
- `-t` specify for which client you want to train LM starting from 0;
- `-b` set this flag to add bias to one client's data (the fourth client by default in our approach)

Example of training an unbiased LM for the first client:
```
python run_CheXpert_LM.py -e 30 -c cardiomegaly -d -t 0
```

Example of training a biased LM for the fourth client:
```
python run_CheXpert_LM.py -e 30 -c cardiomegaly -t 3 -b
```
Note that if data is already distributed, setting a flag `-d` will produce an error. Also, consider renaming or removing the clients' folders for cardiomegaly data before running the script for pleural effusion.

3. Global Model (**GM**). To train a global model via communicating all the learnable parameters, run the [`run_CheXpert_fed.py`](run_CheXpert_fed.py) script with the following arguments:
- `-nc`, `-c`, `-d`, `-ncl`, `-b` as described above;
- `-nr` number of communication rounds, we used 3 for pleural effusion and 4 for cardiomegaly which is close in the total number of training epochs to centralized versions of these models;
- `-agc` set to aggregate the parameters of convolutional layers, True for GM.

Example of training an unbiased GM:
```
python run_CheXpert_fed.py -c effusion -nr 3 -agc
```

Example of training a biased GM:
```
python run_CheXpert_fed.py -c effusion -nr 3 -b -agc
```

4. Personalized Models (**PM**). Training in a federated setting via communicating only the prototypes and weights of the final fully connected layer of `ProtoPNet` results in PM. To this aim, run the [`run_CheXpert_fed.py`](run_CheXpert_fed.py) without setting a flag `-agc`.

Example of training unbiased PM:
```
python run_CheXpert_fed.py -c cardiomegaly -nr 4
```

Example of training biased PM:
```
python run_CheXpert_fed.py -c cardiomegaly -nr 4 -b
```

Do not forget to distribute the data by setting `-d` if you have not done so yet.

### 📊 Evaluation
_____________________________
Note that the values of accuracy output during training are not reliable for CheXpert data due to the test set imbalance. Thus we need to reevaluate the models using balanced accuracy metrics. This can be done using [`CheXpert_evaluation.ipynb`](CheXpert_evaluation.ipynb) notebook. Please, provide the paths to the models in the corresponding cells.

### 🔍 Bias identification in a test image
_________________________________
To find a patch in a test image mostly activated by the prototypes learned by different models, we adapted a script from the [`ProtoPNet's` repository](https://github.com/cfchen-duke/ProtoPNet). To analyze CM and LM models, run [`local_analysis_CM_LM.py`](local_analysis_CM_LM.py) specifying the name of a test image and paths to the models and prototypes. The script will output the predicted class and top-10 most activated prototypes with the similarity scores and class-connection values. It will also create a folder with the most activated patches in the test image and the corresponding nearest prototypes from the training set. The folder is named `most_activated_prototypes`. Consider saving it in a different directory or renaming it before running the script again.

For the GM and PM, run the [`local_analysis_fed.py`](local_analysis_fed.py) script to find the most activated patches in a test image. In this case, you also need to specify the number of a client for which you analyze the model.