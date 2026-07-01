import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        sig_z=[]
        for i in np.nditer(z):
            sig_i=1/(1+ math.exp(-i))
            sig_i=np.round(sig_i, 5)
            sig_z.append(sig_i)
        sig_z=np.array(sig_z)
        return sig_z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu_z=[]
        for i in np.nditer(z):
            relu_i=max(0.0, i)
            relu_i=np.round(relu_i, 5)
            relu_z.append(relu_i)
        relu_z=np.array(relu_z)
        return relu_z
