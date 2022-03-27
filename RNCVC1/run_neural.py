# coding=utf-8

########################
# Configuração inicial #
########################

import warnings

import numpy as np
import random
import matplotlib.pyplot as plt

from rncvc.classifiers.neural_net import NeuralNet

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Permite a recarga automática de arquivos python importados
# dúvidas veja: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
    """ retorna erro relativo """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


##########################################################################
# Criação de um modelo e um conjunto de dados para alguns testes.        #
# Definimos um seed para que seja possivel a conferência dos resultados. #
##########################################################################

# input_size = 4
# hidden_size = 10
# num_classes = 3
# num_inputs = 5
#
#
# def init_toy_model():
#     random.seed(0)
#     np.random.seed(0)
#     return NeuralNet(input_size, hidden_size, num_classes, std=1e-1)
#
#
# def init_toy_data():
#     random.seed(1)
#     np.random.seed(1)
#     X = 10 * np.random.randn(num_inputs, input_size)
#     y = np.array([0, 1, 2, 2, 1])
#     return X, y
#
#
# net = init_toy_model()
# X, y = init_toy_data()

#######################
# forward propagation #
#######################

# scores = net.loss(X, reg=0.1)
# print 'Suas predicoes:'
# print scores, '\n'
# print 'Predicoes corretas:'
# correct_scores = np.asarray([
#     [-0.81233741, -1.27654624, -0.70335995],
#     [-0.17129677, -1.18803311, -0.47310444],
#     [-0.51590475, -1.01354314, -0.8504215],
#     [-0.15419291, -0.48629638, -0.52901952],
#     [-0.00618733, -0.12435261, -0.15226949]
# ])
#
# print correct_scores, '\n'
#
# # A diferenca deve ser pequena. Normalmente < 1e-7
# print 'Diferenca entre sua implementacao e os valores corretos:'
# print np.sum(np.abs(scores - correct_scores))

###########################
# foward: função de custo #
###########################

# loss, _ = net.loss(X, y, reg=0.1)
# correct_loss = 1.30378789133
#
# # deve ser pequena a diferenca, normalmente < 1e-12
# print 'Diferença da sua funcao de custo e do custo correto:'
# print np.sum(np.abs(loss - correct_loss))

############
# backward #
############

# from rncvc.gradient_check import eval_numerical_gradient

# Use o gradiente numérico para verificar sua implementação da etapa backward.
# Se sua implementação estiver correta, a diferença entre os gradientes será
# inferior a 1e-8 para cada um das camadas de pesos: W1, W2, b1, b2.

# loss, grads = net.loss(X, y, reg=0.1)
#
# # as diferenças devem ser pequenas (<1e-8)
# for param_name in grads:
#     f = lambda W: net.loss(X, y, reg=0.1)[0]
#     param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
#     print '%s erro relativo max: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

#################################
# treinando a rede (modelo toy) #
#################################

# net = init_toy_model()
# stats = net.train(X, y, X, y, learning_rate=1e-1, reg=0.1, num_iters=100, verbose=False)
#
# print 'Loss de treinamento: ', stats['loss_history'][-1]
#
# # plotagem dos valores de custo durante o treinamento
# plt.plot(stats['loss_history'])
# plt.xlabel('iteration')
# plt.ylabel('training loss')
# plt.title('Training Loss history')
# plt.show()


#########################
# carregando o CIFAR-10 #
#########################

from rncvc.data_utils import load_CIFAR10, save_model, load_model


def get_CIFAR10_data():
    """
	Carregando o CIFAR-10 e efetuando pré-processamento para preparar os dados
	para entrada na Rede Neural.
	"""
    # Carrega o CIFAR-10
    cifar10_dir = 'rncvc/datasets/cifar-10-batches-py'
    X_train, y_train, X_valid, y_valid = load_CIFAR10(cifar10_dir)

    # Normalizacao dos dados: subtracao da imagem media
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image

    print X_train.shape
    print X_valid.shape

    # Imagens para linhas
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)

    return X_train, y_train, X_valid, y_valid


# Utiliza a funcao acima pra carregar os dados.
X_train, y_train, X_valid, y_valid = get_CIFAR10_data()
print 'Shape dados treinamento: ', X_train.shape
print 'Shape das classes (treinamento): ', y_train.shape
print 'Shape dados validacao: ', X_valid.shape
print 'Shape das classes (validacao): ', y_valid.shape

################################
# treinando a rede no cifar-10 #
################################

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = NeuralNet(input_size, hidden_size, num_classes)

# Treina a rede
stats = net.train(X_train, y_train, X_valid, y_valid,
                  num_iters=1000, batch_size=200,
                  learning_rate=1e-4, learning_rate_decay=0.95,
                  reg=0.5, verbose=True)

# Efetua predicao no conjunto de validacao
val_acc = (net.predict(X_valid) == y_valid).mean()
print 'Acuracia de validacao: ', val_acc

# Salva o modelo da rede treinada
model_path = 'model.bin'
save_model(model_path, net)

# funções de plotagem

# Plota a função de custo e acurácia
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend(loc=3)
plt.show()

from rncvc.vis_utils import visualize_grid


# Visualiza os pesos da rede

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


show_net_weights(net)

# #################################################################################################################### #
# Desafio RNCVC 2016
# Dicas
# Tune os hiperparâmetros (taxa de aprendizado, número de neurônios, etc).
# Use validação cruzada para achar os melhores hiperparâmetros.
# Regularize o modelo: teste o uso de Regularização L2 e Dropout.
# Utilize um esquema mais sofisticado de atualização dos pesos (ex: SGD + Momentum, Nesterov, Adadelta, Adam, RMSProp).
# Salve o melhor modelo obtido usando data_utils.save_model() e data_utils.load_model() para abrir.
# Para o desafio: é fundamental que o trecho abaixo seja executado sem problemas.
# #################################################################################################################### #

# Assuma que o modelo é a classe NeuralNet serializada em disco usando data_utils.save_model()

# best_model = 'path/to/best_model'
# model = load_model(best_model)

# Retorna um vetor de predição (N x 1), onde N é o número de instâncias
# As classes retornadas aqui devem ser inteiros: [0, 1, 2, ..., C]
# Assuma que X_teste e uma matriz de instancias (N_test, D)
# Voce nao tera o X_teste oficial, use outro conjunto para validar.

# predicted_classes = model.predict(X_test)
