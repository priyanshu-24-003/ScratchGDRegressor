import numpy as np

class Multiple_GDRegressor():

    def __init__(self, lr, epochs):
        self.ita = lr
        self.epochs = epochs

        self.all_updated_weights = [self.w.ravel()]
        
    def fit(self, X, Y):
        self.w = np.ones((len(X[0])+1, 1)) # [Bo, B1, B2 ....Bm], Random Initialization of weights
        
        x_1s = np.column_stack((np.ones((len(X), 1)), X)) # adding columns of Ones in place Xi0.
        Y = Y.reshape(-1, 1)
        
        self.all_updated_loss = np.array([self.SSE(x_1s,Y)])

        for e in range(self.epochs):
                        
            gradient_vector = (2 * (x_1s.T)) @ ((x_1s @ self.w) - Y) # slopes/Gradient Vectors
            
            self.w -= self.ita * gradient_vector
            
            self.all_updated_weights.append(list(self.w.ravel()))
            self.all_updated_loss = np.append(self.all_updated_loss, self.SSE(x_1s, Y))

    
    def SSE(self, X, Y):
        
        """
        Loss_Function: Sum of squared error
        X: = x_1s : column vector of ones has been added to Xi0. 
        """
        
        prediction_vector = X @ self.w
        sse = np.sum((Y-prediction_vector) ** 2)

        return sse