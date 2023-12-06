import numpy as np

class KNNClassifier:

    #Se usa un valor de vecinos(k) de 3
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i, sample in enumerate(X_test):

            #Cálculo de distancia euclidiana
            distances = [np.linalg.norm(sample - x) for x in self.X_train]

            #Ordenamiento de índices de menor a mayor según sus distancias, se toman los primeros 3 debido al valor de k, que serían los más cercanos.
            nearest_indices = np.argsort(distances)[:self.k]

            #Se obtienen los valores de clase correspondientes a las instancias más cercanas, usando el índice de estas.
            nearest_labels = [self.y_train[i] for i in nearest_indices]

            #Se evalúa qué valor se repite más, y se asigna como predicciión
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

    #Datos de erntrenamiento (altura y peso, en cm y kg respectivamente)
    X_train = np.array([
        [165, 68], [180, 75], [155, 55], [175, 80], [170, 70],
        [160, 60], [185, 90], [172, 78], [150, 50], [190, 95],
        [178, 72], [163, 58], [188, 85], [174, 68], [200, 110],
        [160, 65], [177, 85], [168, 70], [182, 92], [175, 78],
        [140, 45], [145, 50], [135, 40], [150, 38], [142, 47],
        [168, 75], [182, 88], [170, 72], [177, 82], [165, 68],
        
    ])

    #Valores de clase del conjunto de entrenamiento (edad en años)
    y_train = np.array([27, 32, 24, 35, 29, 
                        22, 40, 33, 20, 45, 
                        34, 26, 38, 30, 50, 
                        26, 39, 32, 43, 36, 
                        18, 21, 17, 23, 19, 
                        30, 36, 31, 38, 28])


    #Prueba
    X_test = np.array([
        [170, 75], [155, 60], [185, 95], [140, 50], [175, 85]
    ])
    #Edades reales: 32, 22, 38, 18, 43

    #Entrenamiento del algoritmo
    knn_classifier = KNNClassifier(k=3)
    knn_classifier.fit(X_train, y_train)

    #Prueba de predicción
    predictions = knn_classifier.predict(X_test)

    #Muestreo de resultados en conjunt
    print("Predicciones:", predictions)
