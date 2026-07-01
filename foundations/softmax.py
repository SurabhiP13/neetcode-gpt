import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        softmax_z=[]
        for i in np.nditer(z):
            numerator=math.exp(i-max(z))
            denom=0
            for j in np.nditer(z):
                denom+=math.exp(j-max(z))
            softmax_z.append(numerator/denom)
        
        return np.round(softmax_z, 4)
