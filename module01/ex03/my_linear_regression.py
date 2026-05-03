import numpy as np
from typing import Optional


class MyLinearRegression:
    """Linear regression model with gradient descent training.
    
    A simple linear regression implementation that uses gradient descent
    to fit a line to the data.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
        """Initialize the linear regression model.
        
        Args:
            thetas: Initial parameters as a numpy array of shape (2, 1)
            alpha: Learning rate for gradient descent (must be positive)
            max_iter: Maximum number of iterations for training
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(thetas, np.ndarray) or thetas.shape != (2, 1):
            raise ValueError("thetas must be a numpy array of shape (2, 1)")
        self.thetas = thetas.copy()
        
        if not isinstance(alpha, (int, float)) or alpha <= 0.0:
            raise ValueError("alpha must be a positive number")
        self.alpha = float(alpha)
        
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        self.max_iter = max_iter

    def fit_(self, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Fit the model using gradient descent.
        
        Args:
            x: Input features as numpy array of shape (m, 1)
            y: Target values as numpy array of shape (m, 1)
            
        Returns:
            The fitted parameters (thetas) or None if fitting fails
        """
        for arr in [x, y]:
            if not isinstance(arr, np.ndarray):
                return None
            if arr.size == 0:
                return None
        m = x.shape[0]
        if x.shape != (m, 1) or y.shape != (m, 1):
            return None
        x_size = x.size
        X_prime = np.c_[np.ones(x_size), x]
        X_prime_T = X_prime.T
        for iteration in range(self.max_iter):
            predictions = X_prime @ self.thetas
            errors = predictions - y
            gradient = (X_prime_T @ errors) / x_size
            if np.allclose(gradient, 0):
                break
            self.thetas = self.thetas - self.alpha * gradient
            print(f"{iteration / self.max_iter * 100:5.2f}%", end="\r")
        return self.thetas

    def predict_(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using the fitted model.
        
        Args:
            x: Input features as numpy array of shape (m, 1)
            
        Returns:
            Predicted values or None if input is invalid
        """
        if not isinstance(x, np.ndarray):
            return None
        m = x.shape[0]
        if m == 0 or x.shape != (m, 1):
            return None
        X = np.c_[np.ones(m), x]
        return X @ self.thetas

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> Optional[np.ndarray]:
        """Compute element-wise loss.
        
        Args:
            y: True values
            y_hat: Predicted values
            
        Returns:
            Element-wise squared errors or None if invalid
        """
        for arr in [y, y_hat]:
            if not isinstance(arr, np.ndarray):
                return None
        m = y.shape[0]
        if m == 0:
            return None
        if y.shape != (m, 1) or y_hat.shape != (m, 1):
            return None
        return (y_hat - y) ** 2

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> Optional[float]:
        """Compute mean squared error loss.
        
        Args:
            y: True values
            y_hat: Predicted values
            
        Returns:
            Mean squared error or None if invalid
        """
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        return float(np.mean(J_elem))


if __name__ == "__main__":

    x = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    print("X :\n", x)
    print("Y :\n", y)

    try:
        lr1 = MyLinearRegression(np.array([[2], [0.7]]))
        print("Class Initialization : Thetas = [[2.0], [0.7]]"
              + " -> y_hat(x) = 2 + 0.7 * x")

        # Example 0.0:
        y_hat = lr1.predict_(x)
        print("y_hat with initial thetas :\n", y_hat)
        # Output:
        # array([[10.74695094],
        #        [17.05055804],
        #        [24.08691674],
        #        [36.24020866],
        #        [42.25621131]])

        # Example 0.1:
        loss_elem = lr1.loss_elem_(y, y_hat)
        print("LOSS_ELEM =", loss_elem)
        # Output:
        # array([[710.45867381],
        #        [364.68645485],
        #        [469.96221651],
        #        [108.97553412],
        #        [299.37111101]])
    except ValueError as e:
        print(f"Error: {e}")

    # Example 0.2:
    loss = lr1.loss_(y, y_hat)
    print("This prediction has a loss of :", loss)
    # Output:
    # 195.34539903032385

    # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1_500_000)

    print("\nTraining the model ... Please wait.\n")

    lr2.fit_(x, y)

    print("After Fit : Thetas = [[{}], [{}]]".format(
        lr2.thetas[0],
        lr2.thetas[1]
        ))
    # Output:
    # array([[1.40709365],
    #        [1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print("Updated y_hat nearest to y :\n", y_hat)
    # Output:
    # array([[15.3408728 ],
    #       [25.38243697],
    #       [36.59126492],
    #       [55.95130097],
    #       [65.53471499]])

    # Example 1.2:
    loss_elem = lr2.loss_elem_(y, y_hat)
    print("LOSS_ELEM =", loss_elem)
    # Output:
    # array([[486.66604863],
    #        [115.88278416],
    #        [ 84.16711596],
    #        [ 85.96919719],
    #        [ 35.71448348]])

    # Example 1.3:
    loss = lr2.loss_(y, y_hat)
    print("Updated loss, nearest to 0 :", loss)
    # Output:
    # 80.83996294128525
