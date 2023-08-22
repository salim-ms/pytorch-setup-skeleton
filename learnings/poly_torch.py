import torch
import math

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

data = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=torch.float)
y = torch.sin(data)
# params
a = torch.randn((), device=device, dtype=torch.float)
print(a)
print(a.shape)
b = torch.randn((), device=device, dtype=torch.float)
c = torch.randn((), device=device, dtype=torch.float)
d = torch.randn((), device=device, dtype=torch.float)

learning_rate = 1e-6
for i in range(2000):
    y_pred = a + b * data + c * data**2 + d * data**3

    loss = (y_pred - y).pow(2).sum().item()

    if i % 100 == 99:
        print(i, loss)
    
    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * data).sum()
    grad_c = (grad_y_pred * data ** 2).sum()
    grad_d = (grad_y_pred * data ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')