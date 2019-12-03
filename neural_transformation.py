from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np
import tensorflow as tf

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
# %matplotlib inline
print(tf.executing_eagerly())
(tf.__version__)


def get_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(3, input_dim=2))
  model.add(tf.keras.layers.Activation("tanh"))
  model.add(tf.keras.layers.Dense(3))
  model.add(tf.keras.layers.Activation("tanh"))
  model.add(tf.keras.layers.Dense(3))
  model.add(tf.keras.layers.Activation("tanh"))
  model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
  model.summary()
  model.compile(optimizer="nadam", loss=tf.keras.losses.BinaryCrossentropy())
  return model

class CallbackWeigths(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch%10 == 0:
      cache = self.forward_prop()
      epochs_dict[epoch] = cache

  def forward_prop(self):
    dataset, lines, ds = start_data
    lines = convert_matrixes_into_data_points(lines[0],lines[1]) #paso ambas matrices 
    to_transform = (dataset, lines, ds)
    cache = {}
    cache_inner = {}
    cache_inner["input"] = to_transform
    for l in self.model.layers[0:-1]:
      m = tf.keras.Sequential()
      m.add(l)
      to_transform = [m.predict(d) for d in to_transform]
      cache_inner[l.name] = to_transform 
      m = None
      del m
    cache["inner_outputs"] = cache_inner
    m = tf.keras.Sequential()
    m.add(self.model.layers[-1])
    cache["output"] = m.predict(to_transform[-1])
    return cache

def get_lines(nb_lines,resolution, dataset):
  x_min,y_min,x_max,y_max = get_limits(dataset)
  inf, sup = np.min([x_min,y_min]), np.max([x_max,y_max])
  ls = np.linspace(inf,sup,nb_lines)
  fill = np.linspace(inf,sup,resolution) 
  xx, yy = np.meshgrid(ls,fill)
  x_coords = np.concatenate((xx, yy), axis=1)
  y_coords = np.concatenate((yy, xx), axis=1)
  #x_cords and y_cords are of shape: resolution x (nb_lines * 2) 
  return x_coords, y_coords

def get_random_data(size,f,to):
  return np.random.random_sample([size,1]) * (to-f) + f

def get_circular_data(nb_points_per_class,f,to):
  length = get_random_data(nb_points_per_class,f,to)
  rad = get_random_data(nb_points_per_class,0,2*np.pi) 
  x = length * np.sin(rad)
  y = length * np.cos(rad)
  return np.concatenate((x, y), axis=1)

def get_circular_dataset(nb_points_per_class):
  dataset = np.concatenate((get_circular_data(nb_points_per_class,0,2), get_circular_data(nb_points_per_class,3,5)), axis=0)
  labels = np.concatenate((np.ones([nb_points_per_class,1],dtype=int), np.zeros([nb_points_per_class,1],dtype=int)), axis=0)
  return dataset, labels

def get_decision_points(dataset, resolution):
  x_min,y_min,x_max,y_max = get_limits(dataset)
  inf, sup = np.min([x_min,y_min]), np.max([x_max,y_max])
  p = np.linspace(inf,sup,resolution)
  xx, yy = np.meshgrid(p,p)
  X = convert_matrixes_into_data_points(xx,yy)
  return X

def generate_all_data(nb_lines = 19, resolution_lines = 100, resolution_decision=200, nb_points_per_class = 50):
  dataset, labels = get_circular_dataset(nb_points_per_class)
  lines = get_lines(nb_lines,resolution_lines,dataset)
  decision_points = get_decision_points(dataset, resolution_decision)
  return dataset, labels, lines, decision_points

def get_limits(dataset):
  x_min, x_max = dataset[:, 0].min(), dataset[:, 0].max()
  y_min, y_max = dataset[:, 1].min(), dataset[:, 1].max()
  return x_min,y_min,x_max,y_max

def convert_matrixes_into_data_points(xx,yy):
  nb_coordinates = xx.shape[0] * xx.shape[1] 
  xx = xx.reshape((nb_coordinates,1))
  yy = yy.reshape((nb_coordinates,1))
  return np.concatenate((xx, yy), axis=1)

def convert_data_points_into_lines(dp, nb_lines, resolution):
  xx = dp[:,0].reshape((resolution, (nb_lines * 2)))
  yy = dp[:,1].reshape((resolution, (nb_lines * 2)))
  return xx, yy

def plot_all(ax, dataset, labels, decision_points, lines, model):
  plot_decision_region(decision_points,model,ax)
  plot_datapoints(ax, dataset, labels, 1)
  plot_axes(ax, lines)

def prepare_figure():
  fig = plt.figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
  return fig

def plot_axes(ax, lines):
  ax.plot(lines[0], lines[1], c="black", linewidth=.5)

def plot_datapoints(ax, datapoints, labels, alpha):
  colors = ["b", "r"]
  x = datapoints[:,0]; y = datapoints[:,1]
  for l in np.unique(labels):
    ix = np.where(labels == l)[0]
    ax.scatter(x[ix], y[ix], c = colors[l], alpha=alpha)

def plot_decision_region(ax, datapoints, prediction, alpha=.1):
  # Plotting decision regions
  classes = get_classes_from_prediction(prediction)
  colors = ["b", "r"]
  x = datapoints[:,0]; y = datapoints[:,1]
  for l in np.unique(labels):
    ix = np.where(classes == l)[0]
    ax.scatter(x[ix], y[ix], c = colors[l], alpha=alpha, linewidths=.0, marker="s")

def plot_decision_region_with_confidence(ax, datapoints, prediction, alpha=.1):
  classes = get_classes_from_prediction(prediction)
  prediction = np.where(prediction>=.5, prediction, 1-prediction)
  colors = [matplotlib.colors.to_rgba("b"), matplotlib.colors.to_rgba("r")]
  alpha_array = alpha * normalize(prediction)  
  rgba_colors = np.zeros((datapoints.shape[0],4))
  for l in np.unique(classes):
    ix = np.where(classes == l)[0]
    rgba_colors[ix] = colors[l] 
  rgba_colors[:, 3] = alpha_array[:,0]
  x = datapoints[:,0]; y = datapoints[:,1]
  for l in np.unique(classes):
    ix = np.where(classes == l)[0] 
    ax.scatter(x[ix], y[ix], c = rgba_colors[ix], linewidths=.0, marker="s")

def sigmoid(x):
  return 1/(1 + np.exp(-x)) 

def softmax(x):
  return np.exp(x)/sum(np.exp(x))

def normalize(x):
  return (x-np.min(x)) / (np.max(x) - np.min(x))

def train_model(model, dataset, labels, nb_epochs, batch_size):
  model.fit(dataset,labels,
            batch_size=batch_size, epochs=nb_epochs,
            verbose=2,
            shuffle=True,
            callbacks=[CallbackWeigths()])
  
def get_classes_from_prediction(prediction):
  return (prediction > .5)

trainables = []
nb_lines = 13; resolution_line = 100; resolution_decision = 200
nb_points_per_class = 64
nb_epochs = 512; batch_size = 32
epochs_dict = {}

dataset, labels, lines, ds = generate_all_data(nb_lines,resolution_line,resolution_decision,nb_points_per_class)
start_data = dataset, lines, ds

model = get_model()
train_model(model, dataset, labels, nb_epochs, batch_size)

def plot_epoch(epoch, with_confidence, layer):
  plt.close()
  fig = prepare_figure()
  axes = fig.subplots(2, 3, sharey=True)
  i = 1
  tam = "11"
  epoch_cache = epochs_dict[epoch]
  inner = epoch_cache.get("inner_outputs")
  output = epoch_cache.get("output")
  ax = fig.add_subplot(tam + str(i))
  ax.set_xticks([])
  ax.set_yticks([])
  dataset, lines, ds = inner.get(layer)
  if with_confidence: 
    plot_decision_region_with_confidence(ax, ds, output, .1)
  else: 
    plot_decision_region(ax, ds, output, .1)
  plot_datapoints(ax, dataset, labels, 1)
  plot_axes(ax, convert_data_points_into_lines(lines, nb_lines, resolution_line))
  i = i+1
  plt.show()


options = [l.name for l in model.layers]
options.insert(0,"input")
interact(plot_epoch,epoch= widgets.IntSlider(min=0, max=nb_epochs, step=10, value=0), with_confidence=[False,True], layer=options)

for l in options:
  plot_epoch(0, True, l)