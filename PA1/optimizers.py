def gradient_descent(lr):
    i = 0
    max_iter = 1000
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_a, grad_W, grad_b, grad_h = backward(y_true, y_pred)
        for key in W:
            self.W[key] = self.W[key] - lr*grad_W[key]
        for key in b:
            self.b[key] = self.b[key] - lr*grad_b[key]
        
