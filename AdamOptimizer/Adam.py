import numpy as np

class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self, grad):
        self.t += 1
        new_params = []
        
        for i, (p, delta) in enumerate(zip(self.params, grad)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat =self.m[i] / (1-self.beta1 ** self.t)
            v_hat = self.v[i] / (1-self.beta2 ** self.t)
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat)+self.epsilon)
            new_params.append(p)
        self.params = new_params
    
    def get_params(self):
        return self.params