import tensorflow as tf
from tensorflow import keras
import math
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
fashion_mnist = keras.datasets.fashion_mnist#loading fashion MNIST dataset
from numpy import array
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0#scaling values from 0 to 1

test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#defining 4 layers with relu ReLU activation function
    keras.layers.Dense(1000,activation=tf.nn.relu),
    keras.layers.Dense(1000,activation=tf.nn.relu),
    keras.layers.Dense(500,activation=tf.nn.relu),
    keras.layers.Dense(200,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Accuracy Percentage=', test_acc)
predictions = model.predict(test_images)
np.argmax(predictions[0])
test_labels[0]

  
arr=model.get_weights()#this will store pruned weight matrix by individual weights
arr1=model.get_weights()#this will store pruned weight matrix neuron wise


#function for weight pruning
def weightprune(percent):
  layer=list()#to store layer where weight is present
  row=list()#to store row where weight is present
  collumn=list()#to store collumn where weight is present
  weight_list=list()
  for ctr in range(0,7,2):
    for i in range(0,len(arr[ctr])):
      for j in range(0,len(arr[ctr][0])):
        weight_list.append(abs(arr[ctr][i][j]))#adding absolute weight values to the list
        layer.append(ctr)
        row.append(i)
        collumn.append(j)
    weight_list,layer,row,collumn=zip(*sorted(zip(weight_list,layer,row,collumn)))#sorting all the weights of neural network with corresponding layer,collumn and row
    weight_list=list(weight_list)
    layer=list(layer)#converting to list
    row=list(row)
    collumn=list(collumn)
    for i in range(0,int((percent/100)*(len(weight_list)))):
      arr[layer[i]][row[i]][collumn[i]]=0#setting bottom k% weights of entire network to 0
  return np.asarray(arr)
  
#   print(arr)



#function for neuron pruning
def neuronprune(percent):
  sum_list=[[0]*1000,[0]*1000,[0]*500,[0]*200]#these numbers change with changing hidden layer configuration
  collumn_list=[[0]*1000,[0]*1000,[0]*500,[0]*200]#to store corresponding collumns to be deleted
  square_sum=0
  for ctr in range(0,7,2):#accessing alternate indexes for weights
    for j in range(0,len(arr1[ctr][0])):
      for i in range(0,len(arr1[ctr])):
        square_sum=square_sum+arr1[ctr][i][j]*arr1[ctr][i][j]
      sum_list[int(ctr/2)][j]=(math.sqrt(square_sum))
      square_sum=0    #to store the squared sum of weights
# print(sum_list)
  for i in range(0,4):#upper limit equal to number of hidden layers
    
    collumn_list[i]=np.argsort(sum_list[i])#sorting the collumns and storing collumn number of sorted in collumn_list
# print(sum_list)
# print(collumn_list)
  
  
  for i in range(0,len(collumn_list)):
    for j in range(0,int((percent/100)*len(collumn_list[i]))):
      layer=2*i
      collumn=collumn_list[i][j]
      for k in range(0,len(arr1[layer])):
        
        arr1[layer][k][collumn]=0#deleting layer wise bottom k% neurons
# print(arr1)
  return np.asarray(arr1)


number=int(input("Number of pruning percentages"))
k=[0]*number

print("Enter percentages(k) for pruning")
for i in range(0,len(k)):
  k[i]=int(input())
for i in range(0,len(k)):
  start=time.time()
  percentage=k[i]
  model.set_weights(neuronprune(percentage))
  #print ("Final Mat seperatlyrix",model.get_weights())
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy for ',percentage,'% pruning using neuron pruning=', test_acc)
  end=time.time()
  print("Time taken=",end-start)
  
  start1=time.time()
  model.set_weights(weightprune(percentage))
  #print ("Final Matrix",model.get_weights())
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy for ',percentage,'% pruning using weight pruning=', test_acc)
  end1=time.time()
  print("Time taken=",end1-start1)