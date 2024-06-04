import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        # Inicializa os pesos e os bias da camada oculta e da camada de saída com valores aleatórios
        self.hidden_weights = np.random.randn(hidden_dim, input_dim)
        self.hidden_bias = np.random.randn(hidden_dim)
        self.output_weights = np.random.randn(output_dim, hidden_dim)
        self.output_bias = np.random.randn(output_dim)
        self.learning_rate = lr

    def _sigmoid(self, x):
        # Função de ativação sigmoide
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # Derivada da função de ativação sigmoide
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Calcula a entrada e a saída da camada oculta
        self.hidden_layer_input = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        self.hidden_layer_output = self._sigmoid(self.hidden_layer_input)

        # Calcula a entrada e a saída da camada de saída
        self.output_layer_input = np.dot(self.output_weights, self.hidden_layer_output) + self.output_bias
        self.output = self._sigmoid(self.output_layer_input)
        return self.output

    def backward_propagation(self, inputs, expected_output):
        # Calcula o erro na camada de saída
        output_error = expected_output - self.output
        output_delta = output_error * self._sigmoid_derivative(self.output)

        # Propaga o erro de volta para a camada oculta
        hidden_error = np.dot(self.output_weights.T, output_delta)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_layer_output)

        # Atualiza os pesos e os bias da camada de saída
        self.output_weights += self.learning_rate * np.dot(output_delta.reshape(-1, 1), self.hidden_layer_output.reshape(1, -1))
        self.output_bias += self.learning_rate * output_delta

        # Atualiza os pesos e os bias da camada oculta
        self.hidden_weights += self.learning_rate * np.dot(hidden_delta.reshape(-1, 1), inputs.reshape(1, -1))
        self.hidden_bias += self.learning_rate * hidden_delta

    def train(self, training_data, training_labels, epochs=10000):
        # Treina a rede neural por um número especificado de épocas
        for _ in range(epochs):
            for inputs, expected_output in zip(training_data, training_labels):
                self.forward_propagation(inputs)
                self.backward_propagation(inputs, expected_output)

    def predict(self, inputs):
        # Realiza uma previsão para os dados de entrada fornecidos
        return self.forward_propagation(inputs)

    def compute_accuracy(self, test_data, test_labels):
        # Calcula a acurácia da rede neural nos dados de teste
        correct_predictions = 0
        for inputs, expected_output in zip(test_data, test_labels):
            prediction = self.predict(inputs)
            if round(prediction[0]) == expected_output[0]:
                correct_predictions += 1
        return correct_predictions / len(test_data)

# Dados de entrada e saída para a porta XOR
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_labels = np.array([[0], [1], [1], [0]])

# Inicializando a rede neural
nn = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1)

# Treinando a rede neural
nn.train(training_data, training_labels)

# Testando a rede neural
for inputs in training_data:
    print(f"Entrada: {inputs} -> Saída prevista: {nn.predict(inputs)}")

# Calculando a acurácia
accuracy = nn.compute_accuracy(training_data, training_labels)
print(f"Acurácia: {accuracy * 100:.2f}%")
