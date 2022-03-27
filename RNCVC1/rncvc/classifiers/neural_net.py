# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


class NeuralNet(object):
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

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
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
                       'W2': std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

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

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape  # instâncias por atributos

        # Calcula a etapa forward
        scores = None
        #############################################################################
        # TODO: Implemente a etapa forward, calculando as predicoes das entradas.   #
        #############################################################################
        
        relu = np.vectorize(lambda x: max(x, 0))

        hidden_layer = relu(np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2

        # Armazene o resultado na variavel scores cujo shape deve ser (N, C).       #
        #############################################################################
        #############################################################################
        #                              FIM DO SEU CODIGO                            #
        #############################################################################

        # Sem passar as classes por parametros retorna
        if y is None:
            return scores

        # Calcula o custo
        loss = None
        #############################################################################
        # TODO: Implemente a etapa forward e calcule o custo. Armazene o resultado  #
        # TODO: na variavel loss (escalar). Use a funcao de custo do Softmax.       #
        #############################################################################

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        probs = np.array(map(softmax, scores))  # probability for each class, for each object
        errors = np.array(
            map(
                lambda i: -np.log(probs[i][y[i]]),
                xrange(N)
            )
        )  # classification error for each instance

        reg_add = (reg / 2. * N) * ((W1 ** 2).sum() + (W2 ** 2).sum())  # add factor in the overall loss
        loss = (1./N) * (
            np.sum(
                errors
            ) + reg_add
        )  # overall loss
        #############################################################################
        #                              FIM DO SEU CODIGO                            #
        #############################################################################

        # Etapa de backward : Calcular os gradientes
        grads = {}
        ###################################################################################
        # TODO: Calcule os gradientes dos pesos e dos biases. Armazene os                 #
        # TODO: resultados no dicionario grads. Por exemplo, grads['W1'] deve armazenar   #
        # TODO: os gradientes relativos a W1, sendo uma matriz do mesmo tamanho de W1.    #
        ###################################################################################

        # softmax derivative
        d_probs = probs.copy()
        d_probs[range(N), y] -= 1.  # subtracts by one only in the correct class probability
        d_probs /= N
        
        # backpropagate to W2 and b2
        d_W2 = np.dot(hidden_layer.T, d_probs)
        d_b2 = np.sum(d_probs, axis=0, keepdims=True)
        d_W2 += reg * W2
        
        grads['W2'] = d_W2
        grads['b2'] = d_b2

        # backpropagate to W1 and b1
        dhidden = np.dot(d_probs, W2.T)
        dhidden[hidden_layer <= 0.] = 0.
        
        d_W1 = np.dot(X.T, dhidden)
        d_b1 = np.sum(dhidden, axis=0, keepdims=True)
        d_W1 += reg * W1
        
        grads['W1'] = d_W1
        grads['b1'] = d_b1

        #############################################################################
        #                              FIM DO SEU CODIGO                            #
        #############################################################################

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

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Carregue um minibatch de instancias em X_batch e y_batch.       #
            #########################################################################
            # prevents an error when running the toy code
            arg_batch = np.random.choice(num_train, size=min(batch_size, num_train), replace=False)
            X_batch = X[arg_batch, :]
            y_batch = y[arg_batch]
            #########################################################################
            #                             FIM DO SEU CODIGO                         #
            #########################################################################

            # Calcule a funcao de custo e os gradientes usando o minibatch atual
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: use os gradientes no dicionario grads para atualizar os         #
            # TODO: parametros da rede (armazenados no dicionario self.params)      #
            # TODO: usando gradiente descendente estocastico.                       #
            #########################################################################
            
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1'][0]
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2'][0]
            #########################################################################
            #                             FIM DO SEU CODIGO                         #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

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
        y_pred = None

        ###########################################################################
        # TODO: Implemente esta funcao. Provavelmente ela sera bastante simples   #
        ###########################################################################
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape  # instâncias por atributos

        # Calcula a etapa forward
        scores = None

        relu = np.vectorize(lambda x: max(x, 0))

        hidden_layer = relu(np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        probs = np.array(map(softmax, scores))  # probability for each class, for each object
        ###########################################################################
        #                              FIM DO SEU CODIGO                          #
        ###########################################################################
        
        y_pred = np.argmax(probs, axis=1)
        
        return y_pred
