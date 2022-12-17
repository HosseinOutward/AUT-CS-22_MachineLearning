'''
    
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    
'''
import numpy as np
from scipy.special import logsumexp
from numpy import linalg as LA
import matplotlib.pyplot as plt


def softmax(z):
    lse = logsumexp(z, axis=1)
    return np.exp(z - (np.ones(z.shape).T * lse).T)  # (np.exp(z).T / np.sum(np.exp(z), axis=1)).T


class LogisticRegression:
    def __init__(self, init_theta=None):
        self.theta = init_theta
        self.JHist = None

    def gradientDescent(self, X, y, theta, n_iter, alpha=0.01):
        # self.JHist = []
        # print("Iteration: ", 1, " Cost: ", self.computeCost(X, y, theta),
        #       " accurecy: ", self.accuracy(X, y, theta))
        for i in range(n_iter):
            # self.JHist.append((self.computeCost(X, y, theta), theta))
            grad=self.computeGradient(X, y, theta)
            # if i % (n_iter / 5) == (n_iter / 5) - 1:
                # print("Iteration: ", i + 1, " Cost: ", self.computeCost(X, y, theta),
                #       " accurecy: ", self.accuracy(X, y, theta))
                # print("average gradiant: ", np.sum(np.abs(grad), axis=0) / grad.shape[0])
            theta -= alpha * grad
        return theta

    def fit(self, X, y, n_iter=100, alpha=0.01):
        self.theta = self.gradientDescent(X, y, self.theta, alpha=0.01, n_iter=n_iter)

    def predict(self, X, theta=None):
        if theta is None: theta = self.theta
        return softmax(X @ theta)

    def computeCost(self, X, y, theta):
        pred = self.predict(X, theta)
        pred = pred.T[y @ np.array(range(y.shape[1])), range(y.shape[0])].T
        return -np.log(pred).sum()

    def computeUncertain(self, pred, tresh=0.5):
        uncert=(pred<tresh).sum(axis=1)-1
        pred=pred.copy().T.tolist()
        pred.append(uncert)
        return np.argmax(pred, axis=0).T

    def computeGradient(self, X, Y, theta):
        gradiant = np.zeros(theta.shape)

        for x, y in zip(X, Y):
            y_arg = np.argmax(y)
            z = x @ theta
            sj = np.exp(z[y_arg] - logsumexp(z))

            dz_dt = np.zeros(theta.shape).T
            dz_dt[y_arg] += x

            dsj_dz = np.zeros(theta.shape[1])
            dsj_dz[y_arg] = sj
            dsj_dz -= sj * np.exp(z - logsumexp(z))

            dJ_dsj = -1 / sj

            temp = np.zeros(theta.shape)
            temp[:, y_arg] += (dJ_dsj * dsj_dz) @ dz_dt
            gradiant = gradiant + temp
        return gradiant

    def kfoldcrossval(self, X, y, k):
        X=np.array(X).copy(); y=np.array(y).copy()
        np.random.shuffle(X); np.random.shuffle(y)
        bin=len(X)//k
        res={
            '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'accuracy': 0,
        }
        count=0
        for i in range(0,len(X),bin):
            X_train=np.concatenate((X[:i],X[i+bin:]))
            y_train=np.concatenate((y[:i],y[i+bin:]))
            X_test=X[i:i+bin]
            y_test=y[i:i+bin]
            model=type(self)(init_theta=np.random.random((X.shape[1], y.shape[1]))*2-1)
            model.fit(X_train,y_train, 30, 1)
            model.fit(X_train,y_train, 70, 0.5)
            model.fit(X_train,y_train, 100, 0.01)

            from sklearn.metrics import classification_report
            l=classification_report(np.argmax(y_test, axis=1),
                        np.argmax(model.predict(X_test), axis=1), output_dict=True)
            res={
                '0': {'precision': res['0']['precision']+l['0']['precision'],
                      'recall': res['0']['recall']+l['0']['recall'],
                      'f1-score': res['0']['f1-score']+l['0']['f1-score'],
                      'support': res['0']['support']+l['0']['support']},

                '1': {'precision': res['1']['precision']+l['1']['precision'],
                      'recall': res['1']['recall']+l['1']['recall'],
                      'f1-score': res['1']['f1-score']+l['1']['f1-score'],
                      'support': res['1']['support']+l['1']['support']},

                'accuracy': res['accuracy']+l['accuracy'],
            }
            count+=1
        res={
            '0': {'precision': res['0']['precision']/count, 'recall': res['0']['recall']/count,
                  'f1-score': res['0']['f1-score']/count, 'support': res['0']['support']/count},
            '1': {'precision': res['1']['precision']/count, 'recall': res['1']['recall']/count,
                  'f1-score': res['1']['f1-score']/count, 'support': res['1']['support']/count},
            'accuracy': res['accuracy']/count,
        }
        return res

    def accuracy(self, X, y, theta):
        pred = self.predict(X, theta)
        pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        return np.sum(pred == y) / y.shape[0]

    def draw_confusion_matrix(self, X, y):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        pred = self.predict(X, self.theta)
        pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        cm = confusion_matrix(y, pred)
        s=sns.heatmap(cm, annot=True, vmin=0)
        s.set(xlabel="Predicted", ylabel="Actual")
        # enlarge plot
        plt.gcf().set_size_inches(12.5, 10)
        plt.show(s)

    def draw_uncertain(self, X, y, tresh):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        pred = self.predict(X, self.theta)

        pred = self.computeUncertain(pred, tresh=tresh)
        y = self.computeUncertain(y, tresh=tresh)

        cm = confusion_matrix(y, pred)
        s=sns.heatmap(cm[:-1], annot=True, vmin=0)
        s.set(xlabel="Predicted", ylabel="Actual")
        # enlarge plot
        plt.gcf().set_size_inches(12.5, 10)
        plt.show(s)
