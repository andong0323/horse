#coding:utf-8

def train_test_split(x, y, prob=0.7, random_state=None):
    '''Split x,y into train set and test set
    
    Arguments:
        x {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int or float.

    Keyword Argument:
        prob {float} -- Train data expected rate between 0 and 1.
        (default: {0.7})
        random_state {int} -- Random seed. (default: {None})

    Returns:
        x_train {list} -- 2d list object with int or float.
        x_test {list} -- 2d list object with int or float.
        y_train {list} -- 1d list object with int 0 or 1.
        y_test {list} -- 1d list object with int 0 or 1.
    '''
    if random_state is not None:
        seed(random_state)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(len(x)):
        if random() < prob:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
    # Make the fixed random_state random again
    seed()
    return x_train, x_test, y_train, y_test

def get_r2(reg, x, y):
    '''Calculate the goodness of fit of regression model.
    
    Arguments:
        reg {model} -- regression model.
        x {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int.
    '''
    y_hat = reg.predict(x)
    r2 = _get_r2(y, y_hat)
    print("Test r2 is %.3f!" % r2)
    return r2

def _get_r2(y, y_hat):
    '''Calculate the goodness of fit.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    '''
    m = len(y)
    n = len(y_hat)
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"
    
    sse = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y, y_hat))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1- sse / sst
    return r2

