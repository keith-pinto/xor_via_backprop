import random, math

class SimpleNeuralNetwork:

    def __init__(self):

        # 1st layer 1st neuron, a.k.a 1st neuron
        self.w1 = random.random()
        self.w2 = random.random()
        self.b1 = 0 # In most cases, bias is initialized to 0. why?

        # 1st layer 2nd neuron, a.k.a 2nd neuron
        self.w3 = random.random()
        self.w4 = random.random()
        self.b2 = 0

        # 2st layer 1st neuron, a.k.a 3rd neuron
        self.w5 = random.random()
        self.w6 = random.random()
        self.b3 = 0

        # Intermediate variables
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def perceptron(self, W1, W2, I1, I2, B):
        return (I1*W1) + (I2*W2) + B

    def forwardpass(self, x1, x2):
        self.z1 = self.perceptron(self.w1, self.w2, x1, x2, self.b1)
        self.a1 = self.sigmoid(self.z1) # a1 means activation of 1st neuron

        self.z2 = self.perceptron(self.w3, self.w4, x1, x2, self.b2)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = self.perceptron(self.w5, self.w6, self.a1, self.a2, self.b3)
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def computeG(self, x, y):
        """
            Functions appended with G (example: computeG) calculate
            gradients
        """
        x1, x2 = x
        self.forwardpass(x1, x2)

        self.da3 = self.costG(y, self.a3)
        self.dz3 = self.activationG(self.a3)

        self.dw5 = self.a1 * self.dz3 * self.da3 
        self.dw6 = self.a2 * self.dz3 * self.da3 
        self.db3 = 1 * self.dz3 * self.da3 
        self.da1 = self.w5 * self.dz3 * self.da3
        self.da2 = self.w6 * self.dz3 * self.da3

        self.dz1 = self.da1 * self.activationG(self.a1)
        self.dw1 = self.dz1 * x1
        self.dw2 = self.dz1 * x2
        self.db1 = self.dz1 * 1

        self.dz2 = self.da2 * self.activationG(self.a2)
        self.dw3 = self.dz2 * x1
        self.dw4 = self.dz2 * x2
        self.db2 = self.dz2 * 1

    def costG(self, original_output, predicted_output):
        return 2 * (original_output - predicted_output)

    def activationG(self, a):
        return a * (1 - a)

    def fit(self, X, Y, epochs=1, learning_rate=1):
        for i in range(epochs):
            dw1, dw2, dw3, dw4, dw5, dw6 = [0]*6
            db1, db2, db3 = [0]*3
            for x, y in zip(X, Y):
                self.computeG(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3

            m = len(X)
            self.w1 -= (learning_rate * (dw1/m))
            self.w2 -= (learning_rate * (dw2/m))
            self.w3 -= (learning_rate * (dw3/m))
            self.w4 -= (learning_rate * (dw4/m))
            self.w5 -= (learning_rate * (dw5/m))
            self.w6 -= (learning_rate * (dw6/m))
            self.b1 -= (learning_rate * (db1/m))
            self.b2 -= (learning_rate * (db2/m))
            self.b3 -= (learning_rate * (db3/m))

            predicted_output = self.predict(X)
            print("Epoch: "+ str(i))
            print("Loss: "+ str(self.averageCost(Y, predicted_output)))

    def predict(self, X):
        Y_pred = []
        for x in X:
            x1, x2 = x
            y_pred = self.forwardpass(x1, x2)
            Y_pred.append(y_pred)
        return Y_pred

    def singleCost(self, original_output, predicted_output):
        return (original_output - predicted_output)**2

    def averageCost(self, original_outputs, predicted_outputs):
        m = len(original_outputs)
        cumv_cost = 0
        for x, y in zip(original_outputs, predicted_outputs):
            cumv_cost += self.singleCost(x, y)
        return (cumv_cost / (2*m))
