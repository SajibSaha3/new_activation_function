
###  Activation Function
The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it.
The purpose of the activation function is to introduce non-linearity into the output of a neuron.
 The activation function of a node in an artificial neural network is a function
that calculates the output of the node based on its individual inputs and their weights.
###
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/272b5de7-2447-4cdc-8d14-fb23a2a23679)

### List of Activations function ###
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/efc1e722-fb67-4bff-93d0-13d0a3902952)

### Rectified Linear Units (ReLU) in Deep Learning. ###

The Rectified Linear Unit is the most commonly used activation function in deep learning models. 
The function returns 0 if it receives any negative input, but for any positive value  
x it returns that value back. So it can be written as  
~~~python
f(x)=max(0,x)
~~~
~~~md
where 0 is the initial state of the value which goes with the value of x
~~~
~~~python
import numpy as np
import matplotlib.pyplot as plt
def relU_activation(feature):
    return np.maximum(0, feature)

feature =np.linspace(-5,5,1000)
#print(f"Feature Dataset:{feature}")
predicted_value = relU_activation(feature)


plt.figure(figsize=(5,3))
plt.plot(feature, predicted_value, label = "Relu activation")
plt.title('Relu activation function')
plt.xlabel("feature")
plt.ylabel('predicted value')
plt.grid(True)
plt.legend()
plt.show()
~~~
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/273de572-8350-4661-929c-2a5180cdc8a1)

###  sigmoid function ###
~~~md
A sigmoid function is a mathematical function with a characteristic "S"-shaped curve or sigmoid curve.
It transforms any value in the domain to a number between 0 and 1.It is Usually used in output layer
of a binary classification. so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.
~~~
### Mtahematics Equation ###
~~~python
1/(1 + e-x)
~~~
### Code implementation ###
~~~python
import numpy as np
import matplotlib.pyplot as plt
def sigmoid_activation_function(feature):
    return 1/(1+np.exp(feature))

feature = np.linspace(-5,5,500)

predicted_value =sigmoid_activation_function(feature)

plt.figure(figsize=(5,3))
plt.plot(feature,predicted_value,label="sigmoid_Activation_function")
plt.title("sigmoid Activation function")
plt.xlabel("feature")
plt.ylabel("predicted value")
plt.grid(True)
plt.legend()
plt.show()
~~~

![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/f1c3f471-e633-47df-8e33-a320feceb74a)
###  tan h function ###
~~~md
The activation that works almost always better than sigmoid function is Tanh function also known as Tangent Hyperbolic function.
It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.
Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes
out be 0 or very close to it, it helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.
~~~
### equation ###
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/7f4a1c17-e07c-4741-ac4b-79522710952c)

### 
code Implementation 
###
~~~python
import numpy as np
import matplotlib.pyplot as plt
def tanh_activation_function(feature):
    return np.tanh(feature)

feature = np.linspace(-5,5,800)
predicted_value = tanh_activation_function(feature)
plt.figure(figsize =(5,3))
plt.plot(feature, predicted_value, label = "Tanh activation function")
plt.title("tanh_activation_function")
plt.xlabel("Feature")
plt.ylabel("prediced value")
plt.grid()
plt.legend()
plt.show()
~~~
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/b080ed41-8fe6-4cb8-800d-dd44c9b13f7d)


### elu Activation function ###
~~~md
ELU, also know as Exponential Linear Unit is an activation function which is somewhat similar to the ReLU with some differences.
Similar to other non-saturating activation functions, ELU does not suffer from the problem of vanishing gradients and exploding gradients.
The ELU has the potential of getting better accuracy than the ReLU.An ELU activation layer performs the identity operation on positive inputs
 and an exponential nonlinearity on negative inputs. The default value of α is 1.
~~~
### equation 
 ~~~python
alpha > 0 is: x if x > 0 and alpha * (exp(x) - 1) if x < 0
~~~
###
### code implementation ###
~~~python
import numpy as np
import matplotlib.pyplot as plt
def elu_activation_function(feature, alpha = 1.0):
    
    return np.where(feature>0, feature, alpha *(np.exp(feature)-1))

feature = np.linspace(-5,5,500)
predicted_value= elu_activation_function(feature)

plt.figure(figsize=(5,3))
plt.plot(feature,predicted_value, label= "elu_avtivation")
plt.title("elu_activation_function")
plt.xlabel("Feature")
plt.ylabel("predicted_value")
plt.grid()
plt.legend()
plt.show()
~~~
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/5488f4a5-e2d6-470e-9cae-6907c64ea1b9)

### softmax activation function ###
~~~md
The softmax activation function transforms the raw outputs of the neural network into a vector of probabilities,
essentially a probability distribution over the input classes.the sigmoid activation function is given by the following equation,
and it squishes all inputs onto the range [0, 1].
~~~
### equation ###

![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/ad6fd6f2-0fd5-4157-8082-4c90732a468d)
###   Implementation ###
~~~python
import numpy as np
import matplotlib.pyplot as plt
def softmax_activation_function(feature, alpha =1.0):
    exp_feature= np.exp(feature - np.max(feature))
    return exp_feature / exp_feature.sum(axis = 0)

feature = np.random.rand(5)*10
softmax_class_probability = softmax_activation_function(feature)
print(f"softmax_class_probability:{softmax_class_probability}")


plt.figure(figsize= (5,3))
plt.bar(range(1,6),softmax_class_probability, tick_label=[f"class{i+1}" for i in range(5)])
plt.title("Softmax Activation Function")
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()
plt.show()
~~~
![image](https://github.com/SajibSaha3/new_activation_function/assets/133786664/7a0f5431-bcde-4637-ac76-62aea7d21ef3)


