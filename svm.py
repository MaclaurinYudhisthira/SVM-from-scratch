import numpy as np

class SVM:
    def __init__(self,learning_rate=.1,regularization_parame=.1,n_iter=1000):
        self.learning_rate=learning_rate
        self.regularization_parame=regularization_parame
        self.n_iter=n_iter
        self.w=None
        self.b=None
        
    def fit(self,X,y):
        
        #creating numpy array
        X=np.array(X)
        y=np.array(y)
        
        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1)

        #zero initialization of weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # printing initial cost
        print(f"Value of cost function before gardient descent: {self.cost(X,y)}")
        
        #gardient descent
        for i in range(self.n_iter):
            for _idx,x_i in enumerate(X):
                
                condition=y[_idx]*(x_i @ self.w - self.b) >= 1
                
                # updating gradients
                if condition:
                    self.w=self.w - self.learning_rate*(2*self.regularization_parame*self.w)
                else:
                    self.w=self.w - self.learning_rate*((2*self.regularization_parame*self.w) - np.dot(x_i, y[_idx]))
                    self.b=self.b - self.learning_rate*y[_idx]
        
        # printing initial cost
        print(f"Value of cost function After {self.n_iter} iterations of gardient descent: {self.cost(X,y)}")
    
    def predict(self,X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx).astype('int32')
    
    def cost(self,X,y):
        Loss=1-y*(X @ self.w - self.b)
        Loss=np.where(0>Loss,0,Loss)
        return np.sum(Loss)/len(X)