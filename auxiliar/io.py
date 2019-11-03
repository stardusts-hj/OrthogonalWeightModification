
"""
Timo Flesch, 2019
"""
import numpy as np
import scipy.io as sio
import pickle
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


def load_pickle(file_name, file_dir):
    with open(file_dir + file_name,'rb') as f:
        pickle.load(f)
    print('loaded file from disc')
    return f


def save_pickle(file_name, file_dir, data):
    with open(file_dir + file_name, 'wb') as f:
        pickle.dump(data, f)
    print('stored file to disc')
