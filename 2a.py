import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class PolinomialRegression:
    def __init__(self, x, y, degree, learning_rate=0.001):
        self.x = create_feature_matrix(x, degree)
        self.y = y
        # self.w = tf.Variable(tf.zeros(degree), dtype=tf.float32)
        self.w = tf.Variable(tf.random.uniform((degree,), 0, 1, dtype=tf.float64))
        self.b = tf.Variable(0.0, dtype=tf.float64)
        self.degree = degree
        self.learning_rate = learning_rate
        # Adam optimizator
        self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Funkcija predikcije.
    def _pred(self, x, w, b):
        w_col = tf.reshape(w, (self.degree, 1))
        hyp = tf.add(tf.matmul(x, w_col), b)
        return hyp


    # Funkcija troška i optimizacija.
    def _loss(self, x, y):
        prediction = self._pred(x, self.w, self.b)

        y_col = tf.reshape(y, (-1, 1))
        mse = tf.reduce_mean(tf.square(prediction - y_col))
        
        return mse



    # Racunanje gradijenta
    def _calc_grad(self, x, y):
        with tf.GradientTape() as tape:
            loss_val = self._loss(x, y)
        
        w_grad, b_grad = tape.gradient(loss_val, [self.w, self.b])

        return w_grad, b_grad, loss_val

    # Pojedinacno izracunavanje gradijenta.
    def _train_step(self, x, y):
        w_grad, b_grad, loss = self._calc_grad(x, y)

        self.adam.apply_gradients(zip([w_grad, b_grad], [self.w, self.b]))

        return loss

    # Trening.
    def fit(self, epochs=100):
        for epoch in range(epochs):
            
            # Stochastic Gradient Descent.
            for i in range(self.x.shape[0]):
                curr_x = self.x[i].reshape((1, self.degree))
                curr_y = self.y[i]

                self._train_step(curr_x, curr_y)

    def get_cost(self):
        avg_cost = 0
        for i in range(self.x.shape[0]):
            curr_x = self.x[i].reshape((1, self.degree))
            curr_y = self.y[i]
            loss_val = self._loss(curr_x, curr_y)
            avg_cost += loss_val

        return avg_cost / self.x.shape[0]
           
    def predict(self):
        predictions = []
        for i in range(self.x.shape[0]):
                curr_x = self.x[i].reshape((1, self.degree))
                predictions.append(self._pred(curr_x, self.w, self.b))
        return predictions

# Pomocna funkcija za crtanje podataka.
def plot_data(x, y, predictions):
    plt.scatter(x, y, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    pred_arr = [elem.numpy().flatten()[0] for elem in predictions[0]] 
    print(pred_arr)
    plt.plot(x, pred_arr, color='red')
    # pred_ar2 = [elem.numpy().flatten()[0] for elem in predictions[1]] 
    # plt.plot(x, pred_ar2, color='green')
    plt.show()

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)



df = pd.read_csv('data/funky.csv', header=None)

scaler = StandardScaler()
raw_x = df[0]
# radimo normalizaciju zbog stepenovanja (6 stepen bi proizveo velike vrednosti)
xs = scaler.fit_transform(raw_x.values.reshape(-1, 1)).flatten().astype('float64')
ys = df[1].astype('float64')

all_costs = []
predictions = []  
max_degree = 1

poly_reg = PolinomialRegression(xs, ys, 2)
poly_reg.fit()
print(poly_reg.get_cost())
# predictions.append(poly_reg.predict())

# for degree in range(1, max_degree+1):
#     poly_reg = PolinomialRegression(xs, ys, degree)
#     poly_reg.fit()
#     all_costs.append(poly_reg.get_cost())
#     predictions.append(poly_reg.predict())

# plot_data(xs, ys, predictions)





