import numpy as np


class KNN(object):
	""" Classificador KNN com distancia L2 (euclidiana) """
	
	def __init__(self):
		pass
	
	def train(self, X, y):
		"""
		Treinamento do k-NN e o armazenamento dos dados de treino.
        Ja implementamos pra voce! :)
    
        Entradas:
        - X: numpy array com shape (instancias, atributos) contendo os dados de treino.
        - y: numpy array com as classes de cada instancia, onde y[i] e a classe para X[i].
        """
		self.X_train = X
		self.y_train = y
	
	def predict_labels(self, X, k=1):
		"""
		Faz a predicao de todas as instancias passadas por parametro.
		Obs: voce pode criar funcoes para auxiliar neste processo.
		
		Entradas:
		- X: numpy array com shape (instancias, atributos) contendo os dados de teste.
		- k: numero dos k-vizinhos mais proximos.
		
		Returns:
		- y: numpy array com a classe de cada instancia de teste,
		onde y[i] e a classe predita da instancia de teste X[i].
		"""
		from collections import Counter
		
		def sq_euclidean(x1, x2):
			return ((x1 - x2) ** 2.).sum()
		
		y_predict = map(
			lambda x1: Counter(self.y_train[np.array(
				map(
					lambda x2: sq_euclidean(x1, x2),
					self.X_train
				)).argsort()[:k]]).most_common()[0][0],
			X
		)
		
		return y_predict
