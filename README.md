# grad

This is a toy implementation of automatic differentiation, created to understand the fundamental concepts. This project is inspired by `micrograd`.

## Usage Example

Below is the definition of the `XOR` class.

```python
class XOR(Module):
    def __init__(self):
        self.hidden_layer = Linear(2,2)
        self.output_layer = Linear(2,1)
        self.activation = Tensor.sigmoid

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x
```

```python
def train(model, x, y, optimizer:Optimizer, loss, epoch):
    loss_history = []
    for i in range(epoch):
        optimizer.zero_grad()
        y_pred = model(x)
        output: Tensor= loss(y_pred, y)
        output.backward()
        optimizer.step()

        loss_history.append(output.item())

        if i % 100 == 0:
            print(f"Epoch {i}: Loss: {output.item()}")
    plot(loss_history, epoch) 
    return model
```


```python
x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_t = np.array([0, 1, 1, 0]).reshape(-1,1)
y_t = Tensor(y_t)

model = XOR()
y_pred = model(x)
optimizer = Optimizer(model.params())
loss = BCELogitsLoss()
epoch = 500

model = train(model,x,y_t, optimizer, loss, epoch)
```

