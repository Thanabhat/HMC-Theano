import theano
import theano.tensor as T
import numpy
from mlp import test_mlp
from DBN import test_DBN


def readJOHMCFF(filePath):
    with open(filePath) as f:
        classSize = int(f.readline().rstrip('\n'))
        classList = []
        data_x = []
        data_y = []

        for i in range(0, classSize):
            classList.append(f.readline().rstrip('\n'))
        dataSize, attrSize = [int(x) for x in f.readline().split()]
        for i in range(0, dataSize):
            attrList = f.readline().rstrip('\n').split()
            data_x.append([float(x) for x in attrList[:attrSize]])
            data_y_list = attrList[attrSize].split("@")
            data_y_tmp = [0] * classSize
            for j in range(0, len(data_y_list)):
                try:
                    ind = classList.index(data_y_list[j])
                    data_y_tmp[ind] = 1
                except ValueError:
                    pass
            data_y.append(data_y_tmp)
        f.close()
    return (data_x, data_y), classList, attrSize, classSize


def load_data(dataset):
    train_set, classList, attrSize, classSize = readJOHMCFF("datasets/" + dataset + "/" + dataset + ".train.johmcff")
    valid_set, _, _, _ = readJOHMCFF("datasets/" + dataset + "/" + dataset + ".valid.johmcff")
    test_set, _, _, _ = readJOHMCFF("datasets/" + dataset + "/" + dataset + ".test.johmcff")


    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval, classList, attrSize, classSize

def save_prediction(dataset, predict, classList):
    with open("datasets/" + dataset + "/" + dataset + ".test.pred.johmcff", "w") as f:
        for i in range(0,len(classList)):
            f.write("@ATTRIBUTE Original-p-"+classList[i]+" numeric\n")
        f.write("\n@DATA\n")
        for i in range(0,len(predict)):
            for j in range(0,len(classList)):
                if j > 0:
                    f.write(",")
                f.write(str(predict[i][j]))
            f.write("\n")
        f.close()


datasetName = "imclef07d"
datasets, classList, attrSize, classSize = load_data(datasetName)
# error, predict = test_mlp(datasets = datasets, n_in=attrSize, n_out=classSize, n_hidden=500, batch_size=20, n_epochs=1000, learning_rate=0.005, L1_reg=0.000, L2_reg=0.000)
error, predict = test_DBN(datasets=datasets, n_ins=attrSize, n_outs=classSize, pretraining_epochs=100, pretrain_lr=0.01, training_epochs=300, finetune_lr=0.002, batch_size=20)
save_prediction(datasetName, predict, classList)
pass