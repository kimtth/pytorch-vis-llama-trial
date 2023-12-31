import torch

# https://qiita.com/awawaInu/items/e173acded17a142e6d02

print(torch.__version__)


# 1) einsum
# Here are some examples of common subscripts and their meanings:

# “i,j”: A matrix with dimensions (i,j)
# “i,j,k”: A tensor with dimensions (i,j,k)
# “i->": A vector with length i
# “i,j->i”: Sum the rows of a matrix to produce a vector of length i
# “i,j->j”: Sum the columns of a matrix to produce a vector of length j
# Output String

# please provide me exmaples how to use pytorch einsum.
# 1. Matrix multiplication
a = torch.randint(0, 10, size=(2, 3), dtype=torch.int32)
b = torch.randint(0, 10, size=(3, 2), dtype=torch.int32)
c = torch.einsum('ij,jk->ik', a, b)
'''
In mathematical notation, this can be written as:

c_ik = a_ij * b_jk

where `c_ik` is the element in the `i`th row and `k`th column of the resulting matrix `c`, 
`a_ij` is the element in the `i`th row and `j`th column of matrix `a`, 
and `b_jk` is the element in the `j`th row and `k`th column of matrix `b`.

tensor([[2, 4, 6],
        [9, 3, 9]], dtype=torch.int32)
tensor([[6, 4],
        [9, 2],
        [1, 7]], dtype=torch.int32)
-------------------
tensor([[ 54,  58],
        [ 90, 105]], dtype=torch.int32)

54 = 2*6 + 4*9 + 6*1

'''
print(a)
print(b)
print('-------------------')
print(c)
print('-------------------')

# 2. Element-wise multiplication
a = torch.randint(0, 10, size=(3, 3), dtype=torch.int32)
b = torch.randint(0, 10, size=(3, 3), dtype=torch.int32)
c = torch.einsum('ij,ij->ij', a, b)
print(a)
print(b)
print('-------------------')
print(c)
print('-------------------')

'''
tensor([[1, 8, 9],
        [1, 2, 4],
        [8, 2, 3]], dtype=torch.int32)
tensor([[0, 3, 4],
        [6, 3, 5],
        [8, 4, 7]], dtype=torch.int32)
-------------------
tensor([[ 0, 24, 36],
        [ 6,  6, 20],
        [64,  8, 21]], dtype=torch.int32)
'''

# 3. Summation
# "i,j->i": Sum the rows of a matrix to produce a vector of length i
a = torch.randint(0, 10, size=(2, 2), dtype=torch.int32)
b = torch.einsum('ij->i', a)
print(a)
print('-------------------')
print(b)
print('-------------------')

'''
tensor([[4, 4],
        [6, 0]], dtype=torch.int32)
-------------------
tensor([8, 6])
'''

# "i,j->j": Sum the columns of a matrix to produce a vector of length j
a = torch.randint(0, 10, size=(2, 2), dtype=torch.int32)
b = torch.einsum('ij->j', a)
print(a)
print('-------------------')
print(b)
print('-------------------')

# 2) roll
# 1. Rolling a 1D tensor
x = torch.tensor([1, 2, 3, 4, 5])
print(x)  # Output: tensor([1, 2, 3, 4, 5]) 
x_roll = torch.roll(x, shifts=2)
print(x_roll)  # Output: tensor([4, 5, 1, 2, 3])

# 2. Rolling a 2D tensor along a specific dimension
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y) # Output: tensor([[1, 2, 3],
         #                 [4, 5, 6],
         #                 [7, 8, 9]])
y_roll = torch.roll(y, shifts=1, dims=0)
print(y_roll)
# Output:
# tensor([[7, 8, 9],
#         [1, 2, 3],
#         [4, 5, 6]])
y_roll_2 = torch.roll(y, shifts=1, dims=1)
print(y_roll_2)
# Output:
# tensor([[3, 1, 2],
#         [6, 4, 5],
#         [9, 7, 8]])

z = torch.tensor([[[1, 2, 9], [3, 4, 8]], [[5, 6, 7], [7, 8, 6]]])
# Output:
# tensor([[[1, 2, 9],
#          [3, 4, 8]],
#
#         [[5, 6, 7],
#          [7, 8, 6]]])
# torch.Size([2, 2, 3])
# 3D, Row, Column
z_roll = torch.roll(z, shifts=(1, -1), dims=(0, 2))
print(z_roll)
# Output:
# tensor([[[6, 7, 5],
#          [8, 6, 7]],
#
#         [[2, 9, 1],
#          [4, 8, 3]]])

# 3. Rolling a tensor in both directions
z = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(z) # Output: tensor([[[1, 2],
         #                  [3, 4]],
         # 
         #                 [[5, 6],
         #                  [7, 8]]])
print(z.shape)
# Output:
# torch.Size([2, 2, 2])
z_roll = torch.roll(z, shifts=(1, -1), dims=(0, 2))
print(z_roll)
# Output:
# tensor([[[6, 5],
#          [8, 7]],

#         [[2, 1],
#          [4, 3]]])
'''
roll of dim0(first dimension) + means down
roll of dim2(third dimension) - means left

the tensor z is being shifted along dimensions 0 and 2.

For dimension 0: The elements are shifted by 1 place in the positive direction. 
This means that each matrix moves one place “down”. 
The last matrix wraps around and becomes the first matrix.

For dimension 2: The elements are shifted by 1 place in the negative direction. 
This means that within each matrix, each column moves one place to the “left”. 
The first column in each matrix wraps around and becomes the last column in that matrix.

As you can see, the first matrix (containing 1, 2, 3, 4) has moved down and become the second matrix in z_roll, 
and the second matrix (containing 5, 6, 7, 8) has wrapped around to become the first matrix. 
Within each matrix, the columns have shifted to the left.
'''
 
# 3) view
import torch

# 1. Reshaping a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_reshaped = x.view(3, 2)
print(x_reshaped)
# Output:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

# 2. Flattening a tensor
y = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
y_flat = y.view(-1)
print(y_flat)
# Output:
# tensor([1, 2, 3, 4, 5, 6, 7, 8])

# 3. Reshaping a tensor with a single dimension
z = torch.tensor([1, 2, 3, 4, 5, 6])
z_reshaped = z.view(6, 1)
print(z_reshaped)
# Output:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

# 4) Squeeze and unsqueeze

# 1. Squeezing a tensor
x = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
# Output:
# tensor([[[1, 2, 3],
#          [4, 5, 6]]])
# Removing dimensions of length 1
x_squeezed = x.squeeze()
print(x_squeezed)
# Output:
# tensor([[1, 2, 3],
#        [4, 5, 6]])

# 2. Unsqueezing a tensor
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
y_unsqueezed = y.unsqueeze(dim=0)
print(y_unsqueezed)
# Output:
# tensor([[[1, 2, 3],
#          [4, 5, 6]]])

# 3. Squeeze dim=0, 1, 2
x = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
print(f'Original tensor: {x}')
print(f'Shape of original tensor: {x.shape}\n')
# Output:
# Original tensor: tensor([[[1, 2, 3],
#                          [4, 5, 6]]])
# Shape of original tensor: torch.Size([1, 2, 3])
# Squeeze dimension 1
x_squeezed = x.squeeze(dim=1)
print(f'Tensor after squeeze dim=1: {x_squeezed}')
print(f'Shape after squeeze dim=1: {x_squeezed.shape}\n')
# Squeeze dimension 2
x_squeezed = x.squeeze(dim=2)
print(f'Tensor after squeeze dim=2: {x_squeezed}')
print(f'Shape after squeeze dim=2: {x_squeezed.shape}\n')
# In this case, since the size of dimension 0 in your tensor x is 1, squeeze(dim=0) will remove this dimension. 
# However, dimensions 1 and 2 have sizes greater than 1, 
# so squeeze(dim=1) and squeeze(dim=2) will not change the tensor.

# Create a tensor with shape (3, 1, 3)
x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
print(f'Original tensor: {x}')
print(f'Shape of original tensor: {x.shape}\n')
# Squeeze dimension 1
x_squeezed = x.squeeze(dim=1)
print(f'Tensor after squeeze dim=1: {x_squeezed}')
print(f'Shape after squeeze dim=1: {x_squeezed.shape}\n')
# Output:
# Original tensor: tensor([[[1, 2, 3]],
#                          [[4, 5, 6]],
#                          [[7, 8, 9]]])
# Shape of original tensor: torch.Size([3, 1, 3])
# Tensor after squeeze dim=1: tensor([[1, 2, 3],
#                                    [4, 5, 6],
#                                    [7, 8, 9]])

# Create a tensor with shape (3, 3, 1)
y = torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
print(f'Original tensor: {y}')
print(f'Shape of original tensor: {y.shape}\n')
# Squeeze dimension 2
y_squeezed = y.squeeze(dim=2)
print(f'Tensor after squeeze dim=2: {y_squeezed}')
print(f'Shape after squeeze dim=2: {y_squeezed.shape}\n')
# Output:
# Original tensor: tensor([[[1],
#                          [2],
#                          [3]],
#                         [[4],
#                          [5],
#                          [6]],
#                         [[7],
#                          [8],
#                          [9]]])
# Shape of original tensor: torch.Size([3, 3, 1])
# Tensor after squeeze dim=2: tensor([[1, 2, 3],
#                                    [4, 5, 6],
#                                    [7, 8, 9]])
# Shape after squeeze dim=2: torch.Size([3, 3])

# Python program to squeeze the tensor in
# different dimensions

# 4. squeeze
# creating the input tensor
input = torch.randn(3,1,2,1,4)
print("Dimension of input tensor:", input.dim())
print("Input tensor Size:\n",input.size())
 
# squeeze the tensor in dimension 0
output = torch.squeeze(input,dim=0)
print("Size after squeeze with dim=0:\n",
      output.size())
 
# squeeze the tensor in dimension 0
output = torch.squeeze(input,dim=1)
print("Size after squeeze with dim=1:\n",
      output.size())
 
# squeeze the tensor in dimension 0
output = torch.squeeze(input,dim=2)
print("Size after squeeze with dim=2:\n",
      output.size())
 
# squeeze the tensor in dimension 0
output = torch.squeeze(input,dim=3)
print("Size after squeeze with dim=3:\n",
      output.size())
 
# squeeze the tensor in dimension 0
output = torch.squeeze(input,dim=4)
print("Size after squeeze with dim=4:\n",
      output.size())
# output = torch.squeeze(input,dim=5) # Error

'''
Dimension of input tensor: 5
Input tensor Size:
 torch.Size([3, 1, 2, 1, 4])
Size after squeeze with dim=0:
 torch.Size([3, 1, 2, 1, 4])
Size after squeeze with dim=1:
 torch.Size([3, 2, 1, 4])
Size after squeeze with dim=2:
 torch.Size([3, 1, 2, 1, 4])
Size after squeeze with dim=3:
 torch.Size([3, 1, 2, 4])
Size after squeeze with dim=4:
 torch.Size([3, 1, 2, 1, 4])
'''

# 5. unsqueeze
# define the input tensor
input = torch.arange(8, dtype=torch.float)
print("Input tensor:\n", input)
print("Size of input Tensor before unsqueeze:\n",
      input.size())
 
output = torch.unsqueeze(input, dim=0)
print("Tensor after unsqueeze with dim=0:\n", output)
print("Size after unsqueeze with dim=0:\n",
      output.size())
 
output = torch.unsqueeze(input, dim=1)
print("Tensor after unsqueeze with dim=1:\n", output)
print("Size after unsqueeze with dim=1:\n",
      output.size())

'''
Input tensor:
 tensor([0., 1., 2., 3., 4., 5., 6., 7.])
Size of input Tensor before unsqueeze:
 torch.Size([8])
Tensor after unsqueeze with dim=0:
 tensor([[0., 1., 2., 3., 4., 5., 6., 7.]])
Size after unsqueeze with dim=0:
 torch.Size([1, 8])
Tensor after unsqueeze with dim=1:
 tensor([[0.],
        [1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.]])
Size after unsqueeze with dim=1:
 torch.Size([8, 1])
'''

# 6. squeeze and unsqueeze -1

'''
The parameter of unsqueeze and squeeze functions are not a number of dimensions to add/remove, 
it tells on which place should one dimension be added/removed. 
The parameter -1 just means in the end, so squeeze(-1) would remove the last dimension and unsqueeze(-1) would add a new dimension after the current last.
'''

a = torch.randn((4, 4, 1))
a.shape               # torch.size([4, 4, 1])
a.squeeze(2).shape    # torch.Size([4, 4]), dimension 2 has been removed
a.squeeze(-1).shape   # torch.Size([4, 4]), last dimension has been removed (same effect)
a.unsqueeze(0).shape  # torch.Size([1, 4, 4, 1]), one new dimension as first
a.unsqueeze(1).shape  # torch.Size([4, 1, 4, 1]), one new dimension as second
a.unsqueeze(3).shape  # torch.Size([4, 4, 1, 1]), one new dimension as fourth
a.unsqueeze(-1).shape # torch.Size([4, 4, 1, 1]), one new dimension as last (same effect)

# 5) sum
# https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be
 x = torch.tensor([
     [1, 2, 3],
     [4, 5, 6]
   ])
print(x.shape)
# torch.Size([2, 3])

torch.sum(x, dim=0)
tensor([5, 7, 9])

torch.sum(x, dim=1)
tensor([6, 15])

# The key to grasp how *dim* in PyTorch and *axis* in NumPy work was this paragraph from Aerin’s article:
# The way to understand the “axis” of numpy sum is that it collapses the specified axis. 
# So when it collapses the axis 0 (the row), it becomes just one row (it sums column-wise).

y = torch.tensor([
     [
       [1, 2, 3],
       [4, 5, 6]
     ],
     [
       [1, 2, 3],
       [4, 5, 6]
     ],
     [
       [1, 2, 3],
       [4, 5, 6]
     ]
   ])

y.shape

torch.Size([3, 2, 3])

torch.sum(y, dim=0)
tensor([[ 3,  6,  9],
        [12, 15, 18]])

torch.sum(y, dim=1)
tensor([[5, 7, 9],
        [5, 7, 9],
        [5, 7, 9]])

torch.sum(y, dim=2)
tensor([[ 6, 15],
        [ 6, 15],
        [ 6, 15]])

