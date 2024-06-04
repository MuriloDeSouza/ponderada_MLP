# ponderada_MLP

# Rede Neural para Portão XOR

Este repositório contém uma implementação simples de uma rede neural do zero usando Python e NumPy. A rede neural é treinada para simular a porta lógica XOR.

***OBS:*** Vale lembrar que essa parte pe para poder ***rodar o código nomeado mlp_normal.py***.

# Contexto

* ***NeuralNetwork Class(Nome da classe usada no código):***
    * __init__: Inicializa os pesos e bias da rede neural e define a taxa de aprendizado.
    * _sigmoid: Função de ativação sigmoide.
    * _sigmoid_derivative: Derivada da função sigmoide, usada para o cálculo do gradiente.
    * forward_*propagation: Calcula a saída da rede neural para uma dada entrada.
    * backward_propagation: Calcula o erro, propaga-o de volta pela rede e atualiza os pesos.
    * train: Treina a rede neural usando os dados de treinamento.
    * predict: Faz previsões para novos dados de entrada.
    * compute_accuracy: Calcula a acurácia da rede neural nos dados de teste.

* ***Training and Testing(Função para treinar os dados de dentrada):***
    * Os dados de entrada e saída para a porta XOR são definidos.
    * A rede neural é inicializada e treinada com esses dados.
    * As previsões são feitas e a acurácia da rede neural é calculada e exibida.

# Requisitos
* python 3.x
* NumPy

# Instalação

1. Clone o repositório em alguma pasta de trabalho sua:
```cmd
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Pode executar o comando abaixo para poder instalar as dependências e o NumPy
```cmd
pip install numpy
pip install -r requirements.txt
```

# Rodando o código
Primeiramente, abra um novo terminal para poder rodar o código que está localizado no mlp_normal.py dentro da pasta "src"
```cmd
ponderada_MLP\src -> "Caminho da pasta para rodar o código"
python3 mlp_normal.py -> "Comando para rodar o código"
```

# Conclusão
Com essas instruções e explicações, você deve ser capaz de configurar, treinar e testar a rede neural para a porta XOR.
Aqui rodamos o código contido dentro do arquivo ***mlp_normal.py***




















# Autor

Esse projeto foi desenvolvido por mim, Murilo de Souza Prianti Silva