import math
import numpy as np
import nnfs

nnfs.init()



# layer_ouput = [[4.8, 1.21, 2.385], 
#                [8.9, -1.81, 0.2],
#                [1.41, 1.051, 0.026]]


# expo_values = np.exp(layer_ouput)
# print (expo_values)


# norm_values = expo_values / np.sum(expo_values, axis=1, keepdims=True)
# print(norm_values)

softmax_output = [0.7,0.1,0.2]

target_output = [1,0 ,0 ]
target_class = 0

loss = -(math.log(softmax_output[0])*target_output[0]+ math.log(softmax_output[1])*target_output[1] + math.log(softmax_output[2])*target_output[2])

print(loss)

# norm_values = expo_values / np.sum(expo_values)
# print(norm_values)
# print(sum(norm_values))

