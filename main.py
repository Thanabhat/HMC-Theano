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
            data_y_tmp = [0.0] * classSize
            for j in range(0, len(data_y_list)):
                try:
                    ind = classList.index(data_y_list[j])
                    data_y_tmp[ind] = 1.0
                except ValueError:
                    pass
            data_y.append(data_y_tmp)
    return (data_x, data_y), classList


def load_data(dataset):
    train_set, classList = readJOHMCFF("datasets_FUN/" + dataset + "/" + dataset + ".train.johmcff")
    valid_set, _ = readJOHMCFF("datasets_FUN/" + dataset + "/" + dataset + ".valid.johmcff")
    test_set, _ = readJOHMCFF("datasets_FUN/" + dataset + "/" + dataset + ".test.johmcff")
    return [train_set, valid_set, test_set], classList


datasets, classList = load_data("church_FUN")