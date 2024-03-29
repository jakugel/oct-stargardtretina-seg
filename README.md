# oct-stargardtretina-seg
Code for the paper "Retinal Boundary Segmentation in Stargardt Disease Optical Coherence Tomography Images Using Automated Deep Learning"

Link: https://tvst.arvojournals.org/article.aspx?articleid=2770902

If the code and methods here are useful to you and aided in your research, please consider citing the paper.

# Dependencies
* Python 3.6.4
* Keras 2.2.4
* tensorflow-gpu 1.8.0
* h5py
* Matplotlib
* numpy

# Training a model (patch-based)
1. Modify *load_training_data* and *load_validation_data* functions in *train_script_patchbased_general.py* to load your training and validation data (see comments in code).
2. Choose one of the following and pass as first training parameter as shown in code:
    * *model_cifar* (Cifar CNN)
    * *model_complex* (Complex CNN)
    * *model_rnn* (RNN) [default]
3. Can change the desired patch size (*PATCH_SIZE*) as well as the name of your dataset (*DATASET_NAME*).
4. Run *train_script_patchbased_general.py*
5. Training results will be saved in the location defined by *parameters.RESULTS_LOCATION*. Each new training run will be saved in a new seperate folder named with the format: 
*<timestamp>_<model name>_<dataset name>*. Each folder will contain the following files:
    * *config.hdf5* (summary of parameters used for training)
    * *stats_epoch#.hdf5* (training and validation results for each epoch up to epoch #)
    * one or more *model_epoch&.hdf5* files containing the saved model at each epoch &
  
# Training a model (semantic)
1. Modify *load_training_data* and *load_validation_data* functions in *train_script_semantic_general.py* to load your training and validation data (see comments in code).
2. Choose one of the following and pass as first training parameter as shown in code:
    * *model_residual_scSE* (Residual U-Net with scSE blocks) [default]
    * *model_residual_5deep* (Residual U-Net with 5 pooling layers)
3. Choose to use augmentations [default] or no augmentations (see relevant parameters in code) as utilised in the paper.
4. Can change the name of your dataset (*DATASET_NAME*).
5. Run *train_script_semantic_general.py*
6. Training results will be saved in the location defined by *parameters.RESULTS_LOCATION*. Each new training run will be saved in a new seperate folder named with the format: 
*<timestamp>_<model name>_<DATASET_NAME>*. Each folder will contain the following files:
    * *config.hdf5* (summary of parameters used for training)
    * *stats_epoch#.hdf5* (training and validation results for each epoch up to epoch #)
    * one or more *model_epoch&.hdf5* files containing the saved model at each epoch &
  
# Evaluating a model (patch-based)
1. Modify *load_testing_data* function in *eval_script_patchbased_general.py* to load your testing data (see comments in code).
2. Specify trained network folder to evaluate.
3. Specify filename of model to evaluate within the chosen folder: *model_epoch&.hdf5*
4. Run *eval_script_patchbased_general.py*
5. Evaluation results will be saved in a new folder (with the name *no_aug_<DATASET_NAME>.hdf5*) within the specified trained network folder. Within this, a folder is created for each evaluated image containing a range of .png images illustrating the results qualitatively as well as an *evaluations.hdf5* file with all quantitative results. A new *config.hdf5* file is created in the new folder as well as *results.hdf5* and *results.csv* files summarising the overall results after all images have been evaluated.
  
# Evaluating a model (semantic)
1. Modify *load_testing_data* function in *eval_script_semantic_general.py* to load your testing data (see comments in code).
2. Specify trained network folder to evaluate.
3. Specify filename of model to evaluate within the chosen folder: *model_epoch&.hdf5*
4. Run *eval_script_semantic_general.py*
5. Evaluation results will be saved in a new folder (with the name *no_aug_<DATASET_NAME>.hdf5*) within the specified trained network folder. Within this, a folder is created for each evaluated image containing a range of .png images illustrating the results qualitatively as well as an *evaluations.hdf5* file with all quantitative results. A new *config.hdf5* file is created in the new folder as well as *results.hdf5* and *results.csv* files summarising the overall results after all images have been evaluated.

# Still to be added
* Instructions / code for ensembling
* Instructions / code for preprocessing data with contrast enhancement (Girard filter)
