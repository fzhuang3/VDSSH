The log files here were created before the code was tidied up and matched to names used in the paper. The hyperparameter names are different, but the settings are as detailed in the paper and are the same as the defaults used in the code here:
With reference to the log of the proposed method,
alpha and gamma here are unused.
beta is the hyper-parameter for the KL divergence here.
The update_list was not used.


The time taken is reduced because the datasets were first converted to hdf5 to improve the IO connection. This is removed in the code here so that the reader does not need to install additional dependencies and allocate about 100GB of space to run it.
