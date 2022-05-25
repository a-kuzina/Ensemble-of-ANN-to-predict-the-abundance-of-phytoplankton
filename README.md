# Ensemble-of-ANN-to-predict-the-abundance-of-phytoplankton
Methods of ANN Ensembling for Phytoplankton Abundance Prediction

The number of the ensemble method used is passed to the main function as a parameter. 0 - k-folds, 1 - bagging, 2 - boosting
The main function specifies the number of iterations of training independent ensembles and the file for recording results. 
The prep_data function loads the dataset and processes it for the selected ensemble method. 
The selected method is used to create base models, trained on the created data. The base models are aggregated using all available agregation methods. 
Learning results of basic models and ensemble models are output to the console and written to the results file. 

If necessary, you can add a rendering of the loss function graphs, as well as a graph comparing the ensemble prediction with the real value of the target attribute
