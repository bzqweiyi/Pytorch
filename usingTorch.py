# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import numpy as np

x = torch.randn(3, requires_grad=True)
print(x**2)
with torch.no_grad():
    print(x**2)
exit()
print(x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
exit()

a = torch.randn(2, 2)

a = ((a * 3) / (a -1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()

print(b)
exit()

x = torch.ones(2, 2, requires_grad=True)
print("x:", x)
y = x + 2
print(y)
print(y.grad_fn)
z = y*y*3
out = z.mean()
print(z, out)
exit()


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
exit()

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(10)
print(a)

b = a.numpy()

print(b)
a.add_(1)
print(a)
print(b)

