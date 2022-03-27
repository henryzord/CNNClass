# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


class MyNeuralNet(object):
    """
    Uma Rede Neural totalmente conectada de duas camadas.

    Dimensoes:
    N: Entrada da rede
    H: Numero de neuronios na camada escondida
    C: Numero de classes

    Treinamento ocorre com a funcao de custo entropia cruzada + softmax.
    Utilize o ReLU como ativacao da primeira camada oculta.

    Em resumo a arquitetura da rede eh:

    entrada - camada totalmente conectada - ReLU - camada totalmente conectada - softmax

    As saidas da segunda camada sao as predicoes para as classes.
    """

    relu = np.vectorize(
        pyfunc=lambda x: float(max(x, 0.)),
        doc='Method for calculating ReLU.\n\nArgs:\n\tx: A single instance.'
    )

    def __init__(self, input_size, hidden_size, output_size, std=1e-4, dropout=0.5, momentum=0.9):
        """

        Inicializacao do modelo. Os pesos iniciais sao pequenos valores aleatorios.
        Valores de bias sao inicializados com zero.
        Pesos e bias sao armazenados na variavel self.params,
        que e um dicionario e tem as seguintes chaves:

        W1: Pesos da primeira camada; shape (D, H)
        b1: Biases da primeira camada; has shape (H,)
        W2: Pesos da segunda camada; shape (H, C)
        b2: Biases da segunda camada; shape (C,)

        Inputs:
        - input_size: Dimensao D dos dados de entrada.
        - hidden_size: Numero de neuronios H na camada oculta.
        - output_size: Numero de classes C.
        """
        self.params = {'W1': std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size),
                       'p': 1. - dropout, 'momentum': momentum
                       }

    @staticmethod
    def softmax(x):
        """
        Method for calculating the softmax activation.
        
        Args:
            x: A single instance.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def loss(self, X, y=None, reg=0.0):
        """
        Calcula a funcao de custo e os gradientes.

        Entradas:
        - X: instancias de entrada (N, D). Cada X[i] e uma instancia de treinamento.
        - y: classes das instancias de treinamento. y[i] e a classe de X[i],
             y[i] e um valor inteiro onde 0 <= y[i] < C.
             Este parametro e opcional: se nao for passado, serao retornados apenas os valores de predicao.
             Passe o parametro se quiser retornar o valor da funcao de custo e os gradientes.
        - reg: regularizacao L2.

        Retorna:
        Se y é None: retorna matriz de predicoes de shape (N, C), onde scores[i, c] e
        a predicao da classe c relativa a entrada X[i].

        Se y nao for None, retorne uma tupla com:
        - loss: valor da funcao de custo para este batch de treinamento
          samples.
        - grads: dicionario contendo os gradientes relativos a cada camada de pesos
          com respeito a funcao de custo; assume as mesmas chaves que self.params.
        """
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        p = self.params['p']
        N, D = X.shape  # instances x attributes
        
        ###########
        # Forward #
        ###########
        
        H1 = self.relu(np.dot(X, W1) + b1)
        D1 = (np.random.rand(*H1.shape) < p) / p
        H1 *= D1
        scores = np.dot(H1, W2) + b2
        
        if y is None:
            return scores
        
        probs = np.array(map(self.softmax, scores))  # probability for each class, for each object
        errors = np.array(
            map(
                lambda i: -np.log(probs[i][y[i]]),
                xrange(N)
            )
        )  # classification error for each instance
        
        reg_add = (reg / 2. * N) * ((W1 ** 2).sum() + (W2 ** 2).sum())  # add factor in the overall loss
        loss = (1. / N) * (
            np.sum(
                errors
            ) + reg_add
        )  # overall loss
        
        ###################
        # Backpropagation #
        ###################
        
        grads = {}

        # softmax derivative
        d_probs = probs.copy()
        d_probs[range(N), y] -= 1.  # subtracts by one only in the correct class probability
        d_probs /= N
        
        # backpropagate to W2 and b2
        d_W2 = np.dot(H1.T, d_probs)
        d_b2 = np.sum(d_probs, axis=0, keepdims=True)
        d_W2 += reg * W2
        
        grads['W2'] = d_W2
        grads['b2'] = d_b2
        
        # backpropagate to W1 and b1
        dhidden = np.dot(d_probs, W2.T)
        dhidden[H1 <= 0.] = 0.
        
        d_W1 = np.dot(X.T, dhidden)
        d_b1 = np.sum(dhidden, axis=0, keepdims=True)
        d_W1 += reg * W1
        
        grads['W1'] = d_W1
        grads['b1'] = d_b1
        
        return loss, grads
    
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Treine uma rede neural usando SGD (Stochastic Gradient Descent)

        Inputs:
        - X: numpy array (N, D) com dados de treinamento.
        - y: numpy array (N,) com as classes. Onde y[i] = c significa que
          X[i] tem a classe c, onde 0 <= c < C.
        - X_val: numpy array (N_val, D) com dados de validacao.
        - y_val: numpy array (N_val,) com as classes da validacao.
        - learning_rate: taxa de aprendizado (escalar).
        - learning_rate_decay: reducao da taxa de aprendizado por epoca.
        - reg: parametro para controlar a forca da regularizacao.
        - num_iters: numero de iteracoes.
        - batch_size: numero de instancias em cada batch.
        - verbose: boolean; se verdadeiro imprime informacoes durante treinamento.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        # Use SGD para otimizar os parametros em self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        c = self.params['momentum']
        
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
            
            # min function prevents an error when running the toy code
            arg_batch = np.random.choice(num_train, size=min(batch_size, num_train), replace=False)
            X_batch = X[arg_batch, :]
            y_batch = y[arg_batch]
            
            # Calcule a funcao de custo e os gradientes usando o minibatch atual
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            
            # TODO original code
            # self.params['W1'] = self.params['W1'] + (-learning_rate * grads['W1'])
            # self.params['b1'] = self.params['b1'] + (-learning_rate * grads['b1'][0])
            # self.params['W2'] = self.params['W2'] + (-learning_rate * grads['W2'])
            # self.params['b2'] = self.params['b2'] + (-learning_rate * grads['b2'][0])

            # TODO momentum code
            self.params['W1'] += c * (-learning_rate * grads['W1'])
            self.params['b1'] += c * (-learning_rate * grads['b1'][0])
            self.params['W2'] += c * (-learning_rate * grads['W2'])
            self.params['b2'] += c * (-learning_rate * grads['b2'][0])
            
            if verbose and it % 100 == 0:
                print 'iteration %03.d / %03.d: loss %.4f' % (it, num_iters, loss)
            
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                # Decay learning rate
                learning_rate *= learning_rate_decay
        
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
    
    def predict(self, X):
        """
        Use os pesos treinados para efetuar predicoes das instancias de teste.
        Para cada instancia faca a predicao dos valores para cada uma das C classes.
        A classe com maior score sera a classe predita.

        Entradas:
        - X: numpy array (N, D) com N D-dimensional instancias para classificar.

        Retorna:
        - y_pred: numpy array (N,) com as classes preditas para cada um dos elementos em X
          Para cada i, y_pred[i] = c significa c e a classe predita para X[i], onde 0 <= c < C.
        """
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape  # instâncias por atributos
        
        hidden_layer = self.relu(np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        
        probs = np.array(map(self.softmax, scores))  # probability for each class, for each object

        y_pred = np.argmax(probs, axis=1)
        
        return y_pred

# ######################### #
# for testing purposes only #
# ######################### #


def main():
    from rncvc.data_utils import load_CIFAR10

    def get_CIFAR10_data():
        """
        Carregando o CIFAR-10 e efetuando pré-processamento para preparar os dados
        para entrada na Rede Neural.
        """
        # Carrega o CIFAR-10
        cifar10_dir = '../datasets/cifar-10-batches-py'
        X_train, y_train, X_valid, y_valid = load_CIFAR10(cifar10_dir)
    
        # Normalizacao dos dados: subtracao da imagem media
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_valid -= mean_image
    
        # print X_train.shape
        # print X_valid.shape
    
        # Imagens para linhas
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_valid = X_valid.reshape(X_valid.shape[0], -1)
    
        return X_train, y_train, X_valid, y_valid
    
    # Utiliza a funcao acima pra carregar os dados.
    X_train, y_train, X_valid, y_valid = get_CIFAR10_data()
    # print 'Shape dados treinamento: ', X_train.shape
    # print 'Shape das classes (treinamento): ', y_train.shape
    # print 'Shape dados validacao: ', X_valid.shape
    # print 'Shape das classes (validacao): ', y_valid.shape
    
    ################################
    # treinando a rede no cifar-10 #
    ################################
    
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10
    net = MyNeuralNet(input_size, hidden_size, num_classes, dropout=0.5)
    
    # Treina a rede
    stats = net.train(
        X_train, y_train,
        X_valid, y_valid,
        num_iters=5000, batch_size=200,  # was 1000 and 200
        learning_rate=1e-4, learning_rate_decay=0.95,
        reg=0.5, verbose=True
    )
    
    # Efetua predicao no conjunto de validacao
    val_acc = (net.predict(X_valid) == y_valid).mean()
    print 'Acuracia de validacao: ', val_acc
    

if __name__ == '__main__':
    main()
