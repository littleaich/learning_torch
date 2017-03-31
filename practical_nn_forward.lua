require 'nn'

-- m = #input units, n = #output units
m = 5
n = 3

lin = nn.Linear(m, n) -- linear module
sig = nn.Sigmoid() -- sigmoid module

--[[ according to my backprop tutorial, the size of the
weight matrix W_l for layer l is (n x m). Here, in torch,
this is the same
--]]
print(lin)
-- {lin} -- look at inside in cl
-- {sig} -- look at inside in cl

print('====== Linear Layer ======= ')
for k, v in pairs(lin) do
	print('----------')
	print('key =', k)
	print(v)
	print('----------')
end


print('====== Sigmoid Layer ======= ')
for k, v in pairs(sig) do
	print('----------')
	print('key =', k)
	print(v)
	print('----------')
end

-- Theta_2 = concatenation of W_2 and b_2
Theta_2 = torch.cat(lin.bias, lin.weight, 2) -- new Tensor
-- gradWeight: dE/dW and gradBias: dE/db
-- the gradients are not zero, lets make them zero
lin:zeroGradParameters()
gradTheta_2 = torch.cat(lin.gradBias, lin.gradWeight, 2)
-- forward pass
x = torch.randn(m)
a_1 = x
z_2 = lin:forward(a_1)
a_2 = sig:forward(z_2)
-- lets calculate the forward pass manually and compare
z_2_manual = Theta_2 * torch.cat(torch.ones(1), a_1, 1)
a_2_manual = z_2_manual:clone():apply( function (z) 
		return 1 / (1 + math.exp(-z)) end )

-- compare a_2 (auto) with a_2_manual
print(a_2) 
print(a_2_manual)

-- initialize MSE loss function
loss = nn.MSECriterion()
-- get the details in cl by typing ? nn.MSECriterion)
-- {loss} -- look at inside in cl
print('====== Loss Layer ======= ')
for k, v in pairs(loss) do
	print('----------')
	print('key =', k)
	print(v)
	print('----------')
end

loss.sizeAverage = false

-- let us calculate the desired output randomly
y = torch.rand(n)

-- forward loss calculation
E = loss:forward(a_2, y)
print('Loss =', E)
-- which is basically calculating the loss in this way
print('Loss (Manual) =', (a_2-y):pow(2):sum())

-- calculate dE/da_2
dE_da_2 = loss:updateGradInput(a_2, y)
print('dE/da_2 ='); print(dE_da_2)

--[[ calculate delta_2 = (dE/da_2) <HADAMARD_PRODUCT> (da_2/dz_2)
we can compute this derivative directly with updateGradInput() function
updateGradInput(input, gradOutput) here for sigmoid module is as follows
sig:updateGradInput(z_2, dE/da_2)
--]]
delta_2 = sig:updateGradInput(z_2, dE_da_2)
print('delta_2 ='); print(delta_2)
delta_2_manual = dE_da_2:clone():cmul(a_2):cmul(1-a_2)
print('delta_2_manual ='); print(delta_2_manual)











