import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i, sample in enumerate(X_test):
            distances = [np.linalg.norm(sample - x) for x in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            most_common = np.bincount(nearest_labels).argmax()
            
            print(f"Predicción para muestra {i + 1}:")
            print(f"  - Características de muestra: {sample}")
            print(f"  - Índices de instancias más cercanas: {nearest_indices}")
            print(f"  - Etiquetas de instancias más cercanas: {nearest_labels}")
            print(f"  - Predicción final: {most_common}\n")
            
            predictions.append(most_common)
        return predictions

#Ejecución
if __name__ == "__main__":

    '''
    #Datos de entrenamiento
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 0, 1, 1])

    #Datos de prueba
    X_test = np.array([[1, 2], [5, 6]])
    '''

    #Entrenamiento
    X_train = np.array([
        [160, 60], [175, 70], [155, 50], [180, 80], [165, 65],
        [150, 45], [185, 90], [170, 75], [140, 40], [195, 100],
        [180, 70], [165, 50], [190, 85], [175, 65], [200, 110],
        [155, 60], [180, 90], [170, 65], [185, 95], [175, 80],
        [125, 25], [130, 30], [110, 22], [140, 18], [135, 27]
    ])
    y_train = np.array([25, 30, 22, 35, 28, 20, 40, 32, 18, 45, 33, 27, 38, 29, 50, 26, 38, 31, 43, 36, 5, 8, 4, 7, 6])

    #Prueba
    X_test = np.array([
        [170, 68], [160, 55], [190, 95], [155, 48], [90, 14], [140, 33], [160, 50]
    ])
    #Edades reales: 32, 28, 42, 23, 3, 10, 14

    #Entrenamiento del algoritmo
    knn_classifier = KNNClassifier(k=3)
    knn_classifier.fit(X_train, y_train)

    #Prueba de predicción
    predictions = knn_classifier.predict(X_test)

    #Muestreo de resultados en conjunto
    print("Predicciones:", predictions)
