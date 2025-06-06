import random
class Neuron:
  def __init__(self, nin):
    # nin = number of inputs
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def parameters(self):
    return self.w + [self.b]

  def __call__(self, x):
    # this is wild; you can call an instance of Neuron as n(x) :O
    # x are the inputs, we are taking a dot product:
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # sum can take an optional starting value: self.b
    out = act.tanh()
    return out

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
