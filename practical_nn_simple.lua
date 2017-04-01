--[[ The code is taken from the tutorial
https://www.youtube.com/watch?v=atZYdZ8hVCw&index=6&list=PLLHTzKZzVU9ebuL6DCclzI54MrPNFGqbW

However, the comments are written by me - Shubhra Aich (s.aich.72@gmail.com)
--]]

require 'nn'

-- m = #input units, n = #output units
m = 5
n = 3

-- % ================ Layer Definitions Started =============== %
lin = nn.Linear(m, n) -- linear module
sig = nn.Sigmoid() -- sigmoid module

--[[ according to my backprop tutorial, the size of the
weight matrix W_l for layer l is (n x m). Here, in torch,
this is the same
--]]
print(lin)
-- {lin} -- look at inside in cl
-- {sig} -- look at inside in cl

-- the gradients are not zero, lets make them zero
lin:zeroGradParameters()

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
-- % ================ Layer Definitions End =============== %


-- Theta_2 = concatenation of W_2 and b_2
Theta_2 = torch.cat(lin.bias, lin.weight, 2) -- new Tensor
-- gradWeight: dE/dW and gradBias: dE/db

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
print('a_2 ='); print(a_2) 
print('a_2 (manual) ='); print(a_2_manual)

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

--[[ accGradParameters(input, gradOutput, scale) computes the gradients w.r.t. the module parameters. It takes three arguments.
for example, we are using linear module here. So, the input to the module
is a set of inputs x or a_1. The output is the linear combination z_2,
where z_2 = w_2 * a_1 + b_2 or z_2 = Theta_2 * [1, a_2].
So, gradOutput is dE/dz_2 or partial derivative w.r.t. the output of the
module. This is nothing but delta_2 here. 
Finally, it calculates the gradients dE/dw_2 and dE/db_2
--]]
-- updates lin.gradBias and lin.gradWeight
lin:accGradParameters(a_1, delta_2) 
-- lets check
gradTheta_2 = torch.cat(lin.gradBias, lin.gradWeight, 2)
--[[ view calculates transpose actually. we cannot use :t() or :tranpose()
because this is a 1D vector. To perform transpose using those 2 methods, 
we need at least 2 dimensions.

Also, for delta_2, we need to use view just to make it 2 dimensional because in torch multiplication between 1D and 2D tensors are not yet
supported. 
:view(-1,1) means #columns = 1 and use as many rows as needed.
--]]
gradTheta_2_manual = delta_2:view(-1, 1) 
			* torch.cat(torch.ones(1), a_1, 1):view(1, -1)
print('gradTheta_2 ='); print(gradTheta_2)
print('gradTheta_2_manual ='); print(gradTheta_2_manual)

--[[ now lets compute dE/da_1, which in this simple network is not useful,
because we are not going to use this gradient for backprop. However, we 
will need this computation if the network has multiple layers.

updateGradInput(input, gradOutput) computes the gradient of the module
w.r.t. its own input. This is returned in gradInput. So, the gradInput
state variable is updated accordingly.
In our linear module case, it computes dE/da_1 or dE/dx and the function is given by updateGradInput(a_1, dE/dz_2) or updateGradInput(a_1, delta_2)

In our single layer case, we do not want to update 
--]]

-- lin_gradInput = lin:updateGradInput(a_1, delta_2)
lin:updateGradInput(a_1, delta_2)
-- lets check manually
lin_gradInput_manual = lin.weight:t() * delta_2
print('dE/da_1 or dE/dx (auto) ='); print(lin.gradInput)
print('dE/da_1 or dE/dx (manual) ='); print(lin_gradInput_manual)

--[[ until now, we check the backprop calculation in details. However,
torch provides Containers to facilitate and automate the computations in
fewer steps as follows.
--]]
net = nn.Sequential()
net:add(lin) -- add linear layer lin defined above
net:add(sig) -- add sigmoid layer sig defined above
print(net)
--[[ layer gradients are initialized by non-zero values
make them zero before starting training, otherwise they will add up
--]]
net:zeroGradParameters()
pred = net:forward(a_1) -- forward step 
err = loss:forward(pred, y) -- calculate error
--[[ now we backpropagate the error through the loss layer. If you take a look at the nn module backward() function, the calling syntax is given by [gradInput] backward(input, gradOutput)
However, for the loss or criterion layers, the calling syntax is 
backward(predicted_output, target_output). 
If you call backward this way for the loss or Criterion layer, it gives you dE/da_L or dE/dpred where a_L or pred is the output of the final or output layer. L is the final layer according to the backprop tutorial.
--]]
gradCriterion = loss:backward(pred, y) -- computes dE/da_L or dE/dpred
--[[ now call backward(input, gradOutput) for the whole network.
We call net:backward(x, dE/da_L) which updates all the gradients in the network.
--]]
net:backward(a_1, gradCriterion) 
--[[ now, we have calculated all the gradients of the network in a single forward-backward pass. It is time to update the parameters with these gradients according to the gradient-descent algorithm or any of its variant. Remember, the equation of the gradient-descent algorithm is given by w(t+1) = w(t) - learning_rate * dE/dw(t)
We will update the parameters using updateParameters(learningRate) function. 
--]]
learningRate = 0.01
net:updateParameters(learningRate) 

-- lets check the updated parameters manually
-- concat dE/db_2 and dE/dW_2
dE_dTheta_2 = torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2)
print('Updated Parameters (Auto) =')
print(torch.cat(net:get(1).bias, net:get(1).weight, 2)) 
print('Updated Parameters (Manual) =') 
print(Theta_2 - learningRate * dE_dTheta_2)













