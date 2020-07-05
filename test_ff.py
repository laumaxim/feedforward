import feedforward

def test_sigmoid():
    '''unit test of function sigmoid()'''
    a = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    expected = np.array([0.0, 0.27, 0.5, 0.73, 1.0])
    assert np.all(sigmoid(a).round(2) == expected)

def test_feed_forward():
    X_bias = np.hstack((X,np.ones((50,1))))
    hidden_weights = np.random.uniform(size=(3,2))
    outer_weights = np.random.uniform(size=(3,1))
    out = feed_forward(X_bias, [hidden_weights, outer_weights])
    assert out[0].shape == (50, 2)
    assert out[0].shape == (50, 1)

    Xref = np.array([[1.0, 2.0, 1.0]])
    whidden = np.array([[1.0, 2.0, 0.0],
                    [-1.0, -2.0, 0.0]
                        ]).T
    wout = np.array([1.0, -1.0, 0.5]).T

    out = feed_forward(Xref, [whidden, wout])
    assert np.all(out[0].round(2) == np.array([[0.99, 0.01]]))
    assert np.all(out[1].round(2) == np.array([[0.82]]))

def test_logloss():
    ytrue = np.array([0.0, 0.0, 1.0, 1.0])
    ypred = np.array([0.01, 0.99, 0.01, 0.99])
    expected = np.array([0.01, 4.61, 4.61, 0.01])
    assert np.all(log_loss(ytrue, ypred).round(2) == expected)