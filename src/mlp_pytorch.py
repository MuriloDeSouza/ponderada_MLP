import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Inicializa a superclasse nn.Module
        super(NeuralNet, self).__init__()
        # Define a camada oculta
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        # Define a camada de saída
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Define a função de ativação sigmoide
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Propaga os dados pela camada oculta e aplica a função de ativação
        hidden_output = self.activation(self.hidden_layer(x))
        # Propaga os dados pela camada de saída e aplica a função de ativação
        final_output = self.activation(self.output_layer(hidden_output))
        return final_output

def train_network(model, loss_function, optimizer, inputs, labels, epochs=10000):
    for epoch in range(epochs):
        # Zera os gradientes dos parâmetros do modelo
        optimizer.zero_grad()
        # Realiza a previsão para as entradas fornecidas
        predictions = model(inputs)
        # Calcula a perda (erro) entre as previsões e os rótulos verdadeiros
        loss = loss_function(predictions, labels)
        # Propaga o erro de volta pela rede
        loss.backward()
        # Atualiza os pesos do modelo
        optimizer.step()

def evaluate_accuracy(model, inputs, labels):
    # Desabilita o cálculo de gradientes
    with torch.no_grad():
        # Realiza a previsão para as entradas fornecidas
        predictions = model(inputs)
        # Converte as previsões em valores binários (0 ou 1)
        predicted_labels = (predictions > 0.5).float()
        # Conta quantas previsões estão corretas
        correct_predictions = (predicted_labels == labels).float().sum()
        # Calcula a acurácia dividindo o número de previsões corretas pelo total de exemplos
        return correct_predictions / labels.size(0)

# Dados de entrada e saída para a porta XOR
training_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
training_targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Inicializando o modelo de rede neural
input_dim = 2
hidden_dim = 2
output_dim = 1
model = NeuralNet(input_dim, hidden_dim, output_dim)

# Definindo a função de perda e o otimizador
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinando a rede neural
train_network(model, loss_function, optimizer, training_data, training_targets)

# Testando a rede neural
with torch.no_grad():
    for input_data in training_data:
        prediction = model(input_data)
        print(f"Entrada: {input_data.numpy()} -> Saída prevista: {prediction.numpy()}")

# Calculando a acurácia
accuracy_value = evaluate_accuracy(model, training_data, training_targets)
print(f"Acurácia: {accuracy_value.item() * 100:.2f}%")
