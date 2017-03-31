-- this is my first torch code

t = torch.Tensor(2,3,4) -- not initialized
print(t) 

--[[ tensor indexing is in row-major order.
So, for the previous initialization, the 
indexing is in the order K x H x W
where K = number of channel
H = Height (top to bottom) (row)
W = Width ( left to right) (column)
----
In row-major order, a single row is traversed first
you will understand indexing with the following apply()
function. It traverses each of the elements of the 
tensor and intiialize them with their corresponding
linear indices.
--]]

i=0
t:apply(function () i=i+1; return i; end)
print(t)

print(torch.type(t)) -- get data type of t
--[[ the following data types are available
ByteTensor, CharTensor, ShortTensor, IntTensor,
LongTensor, FloatTensor, DoubleTensor
----
The default is DoubleTensor.However, in deep learning, 
we don't need double precision. So, it is better to 
setup the default as FloatTensor. 
--]]
torch.setdefaulttensortype('torch.FloatTensor') -- set default
print(torch.type(torch.Tensor(1,2,3)))
-- we can specify specific type of tensor as follows
print(torch.type(torch.ByteTensor(1,2,3)))

r = torch.DoubleTensor(t):resize(3,8) -- resize tensor t
print(r)

r:zero()
print(r)
print(t)
-- because r and t are pointing to the same references
s = t
s:resize(4,6)
print(s)
print(t)
--[[ s and t have the same reference. So, any modification
in one tensor will affect the other. no deepcopy occurs.
to deepcopy, use clone() as follows
--]]
u = t:clone() -- deep clone
u:random()
print(u)
print(t)






