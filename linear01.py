import numpy as np

x= np.array([[92,4],[123,5],[114,6],[98,4],[99,6],[67,4],[110,6],[78,4],[115,6]])
y = np.array([250,209.5,349.5,250,419,225,549.5,240,340]) #the dataset is really small but will implemet it soon with a larger dataset
w1=-0 #if there are any suggestions to make this model better or implement in a better way please mail it to me @madanprtyaksh@gmail.com will be really thankful of your help/suggestion
w2 =0 
b=-0
learning_rate = 0.00001
trials = 5000000



for i in range(trials):
    w = np.array([w1,w2])
    f = np.dot(x,w) +b
    grad_w1 = 2/len(x)*(np.sum((f-y)*x[:,0]))
    grad_w2 = 2/len(x)*(np.sum((f-y)*x[:,1]))
    grad_b = 2/len(x)*(np.sum(f-y))
    #update parameters
    w1 = w1 - learning_rate*grad_w1
    w2 = w2 - learning_rate*grad_w2
    b = b - learning_rate*grad_b
    #calculate cost
    cost = (np.sum((f - y)**2))
    if i%10000 == 0:
        print(f"cost: {cost} w1 = {w1} w2 = {w2} b = {b}")








# print(f)
# print(cost)
# size = int(input("Size: "))
# rooms = int(input("Rooms: "))
# price = w1*size + w2*rooms + b
# print(price)
