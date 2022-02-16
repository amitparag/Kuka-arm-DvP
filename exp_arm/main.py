import torch
from pathlib import Path
import pickle

from critic import Critic
from learner import Learner
from datagen import make_training_dataloader
import os

from numpy.lib import utils

from utils import path_utils

def establish_baselines():

    ##################################### HYPER PARAMS FOR THE EXP
    NX                  =   14
    EPISODES            =   10
    EARLY_STOP_V        =   1.0e-6
    EARLY_STOP_VX       =   1.0e-6 
    N_EPOCHS            =   1000000
    LR                  =   1.0e-3
    BATCHSIZE           =   64
    TRAIN_SIZE          =   400
    TEST_SIZE           =   100
    NR                  =   3
    NH                  =   3
    NHU                 =   64
    HORIZON             =   200
    

    ############################################################################################# LOGGER
    resultspath = path_utils.results_path()
    logdir = os.path.join(resultspath, f"trained_models/dvp/Order_{1}/Horizon_{HORIZON}/")

    Path(logdir).mkdir(parents=True, exist_ok=True)

    ### Create tag to identify the exp based on horizon
    tag     =   f'horizon_{HORIZON}.pkl'
    INFO    =   {"DVP":[]}
    logs   =   logdir+tag
    with open(logs, "wb") as f:     ### To store logs
        pickle.dump(INFO, f)

    f.close()

    ################################################################################################

    ### Get test data
    test_data   =   torch.load(os.path.join(resultspath, "test_data/test_data.pth"))


    ################################################################################################





    critic = Critic(nx=NX,nr=NR,nh=NH,nhu=NHU)

    learner = Learner(critic=critic,
                    sobolev_order=1,
                    validation_data=test_data,
                    early_stop_v=EARLY_STOP_V,
                    early_stop_vx=EARLY_STOP_VX,
                    n_epochs=N_EPOCHS,
                    lr=LR,
                    create_anchor=True,
                    savePath=logdir)

    ########################################################################################################

    logger = {}

    print(" [INFO] Started Training ..\n ")

    for episode in range(EPISODES):
        print(f" [INFO] Episode {episode}")

        if episode == 0:
                    ################ EPS 0 training data
            dataloader  =   make_training_dataloader(critic=None,horizon=HORIZON,nb_samples=TRAIN_SIZE)
                                                                        


        else:
            assert critic is not None
            dataloader  =   make_training_dataloader(critic=critic,horizon=HORIZON,nb_samples=TRAIN_SIZE)


        critic,info_dict   =   learner.learn(training_data=dataloader,episode=episode, 
                                            name_prefix=f"eps_{episode}")

        logger[f"Episode_{episode}"]  = info_dict
        



        ## CREATE NEW TRAINING DATA
        del dataloader ## Delete previous dataloader

    with open(logs,"wb") as f:
        pickle.dump(logger,f)
    f.close()
    print("Over ...")

establish_baselines()
