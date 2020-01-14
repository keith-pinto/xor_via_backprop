from SimpleNeuralNetwork import SimpleNeuralNetwork

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

snn = SimpleNeuralNetwork()
snn.fit(X, Y, 1, 0.5)

print(snn.predict(X))
#print(snn.averageCost([0, 1, 1, 0], [0, 1, 1, 0]))
