
import numpy as np
import scipy.io as sio

def save_log(log_dict, modelName, logDir, isPy3):
    """
    saves log on harddisk
    """
    fileName = logDir + modelName
    # with open(fileName, 'wb') as f:
    #     if isPy3:
    #         pickle.dump(log_dict,f,protocol=2)
    #     else:
    #         pickle.dump(log_dict,f)

    sio.savemat(fileName, log_dict)
