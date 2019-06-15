import os
import cloudpickle
import numpy as np
import tensorflow as tf

def _save_to_file(save_path, data=None, params=None):
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".pkl"

        with open(save_path, "wb") as file_:
            cloudpickle.dump((data, params), file_)
    else:
        # Here save_path is a file-like object, not a path
        cloudpickle.dump((data, params), save_path)


def save(sess, save_path, data={}):
    params = find_trainable_variables("model")

    params = sess.run(params)

    _save_to_file(save_path, data=data, params=params)



def _load_from_file(self,load_path):
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".pkl"):
                load_path += ".pkl"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

        with open(load_path, "rb") as file:
            data, params = cloudpickle.load(file)
    else:
        # Here load_path is a file-like object, not a path
        data, params = cloudpickle.load(load_path)

    return data, params


def load(sess, load_path, env=None, **kwargs):
    data, params = _load_from_file(load_path)

    # model = cls(None, env, _init_setup_model=False)
    # model.__dict__.update(data)
    # model.__dict__.update(kwargs)
    # model.set_env(env)
    # model.setup_model()
    params1 = find_trainable_variables("model")
    restores = []
    print(np.asarray(params1).shape)
    print(np.asarray(params).shape)
    for param, loaded_p in zip(params1, params):
        restores.append(param.assign(loaded_p))
    sess.run(restores)
    #return model


def find_trainable_variables(key):
    """
    Returns the trainable variables within a given scope

    :param key: (str) The variable scope
    :return: ([TensorFlow Tensor]) the trainable variables
    """
    
    return tf.trainable_variables()