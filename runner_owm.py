import tensorflow as tf
if int(tf.VERSION[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data
# import custom model
from trainer import train_nnet
from nnet import NNet_OWM
# import custom helper functions
from auxiliar.data import gen_splitMNIST


# ignore warning messages (remove clutter from stdout)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

FLAGS = tf.app.flags.FLAGS


# directories
tf.app.flags.DEFINE_string('data_dir',  './data/',
                           """ (string) data directory           """)

# tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints/',
#                             """ (string) checkpoint directory    """)
# log_dir is already defined by abseil, as of r1.14 (fucking hell...)
tf.app.flags.DEFINE_string('logging_dir',          './log/',
                           """ (string) log/summary directory    """)


# dataset
tf.app.flags.DEFINE_integer('n_inputs', 28*28,
                           """ (int) number of inputs """)

tf.app.flags.DEFINE_integer('n_classes', 10,
                           """ (int) number of output classes """)




# model
tf.app.flags.DEFINE_string('model',                'owm',
                            """ (string)  chosen model          """)


tf.app.flags.DEFINE_string('nonlinearity',       'relu',
                            """ (string)  activation function   """)

tf.app.flags.DEFINE_integer('dim_hidden',             800,
                            """ (int) dimensionality of hidden layers """)



# training
tf.app.flags.DEFINE_list('lr',     [[0.2]],
                            """ (list)   learning rate array             """)

tf.app.flags.DEFINE_list('alpha',     [[0.9, 0.6]],
                            """ (list)  alpha array (note:[0]*.001**lambda)  """)

tf.app.flags.DEFINE_float('L2_lambda',            1e-4,
                            """ (float) strength of regulariser """)

tf.app.flags.DEFINE_string('optimizer',       'Momentum',
                            """ (string)   optimisation procedure     """)

tf.app.flags.DEFINE_float('momentum',     0.9,
                            """ (float) momentum strength   """)

tf.app.flags.DEFINE_integer('n_epochs',   20,
                            """ (int)    number of epochs on entire dataset """)

tf.app.flags.DEFINE_integer('display_step',         100,
                            """(int) step size for log to stdout      """)

tf.app.flags.DEFINE_integer('batch_size',         128,
                            """ (int)     training batch size         """)


def main(argv=None):

    FLAGS = tf.app.flags.FLAGS

    # create datasets for the two tasks
    raw_data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    dataset_1 = gen_splitMNIST(raw_data, [0, 1]) # 0 to 1
    dataset_2 = gen_splitMNIST(raw_data, [2, 3]) # 2 to 3
    dataset_3 = gen_splitMNIST(raw_data, [4, 5]) # 4 to 5
    dataset_4 = gen_splitMNIST(raw_data, [6, 7]) # 6 to 7
    dataset_5 = gen_splitMNIST(raw_data, [8, 9]) # 8 to 9
    # tasks = [dataset_1, dataset_2]
    tasks = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    nnet = NNet_OWM()
    train_nnet(tasks, nnet)


if __name__ == '__main__':
    """ take care of flags on load """
    tf.compat.v1.app.run()
