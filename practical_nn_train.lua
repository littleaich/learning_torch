--[[ The code is taken from the tutorial
https://www.youtube.com/watch?v=atZYdZ8hVCw&index=6&list=PLLHTzKZzVU9ebuL6DCclzI54MrPNFGqbW

However, most of the comments are written by me - Shubhra Aich (s.aich.72@gmail.com)

This code does not produce any output. Prior to reading this code, 
read and understand practical_nn_simple.lua step-by-step.
--]]

require 'nn'

--[[ X: feature matrix -> N x D where N = #samples and D = #dimension
Y: target labels -> N x K where N = #samples and K = #classes
--]]

--[[ Stochastic Gradient Descent (SGD) with single sample
Remember, SGD in torch actually updates the parameters for single sample.
--]]
for i = 1, N do
	local pred = net:forward(X[i])
	local err = loss:forward(pred, Y[i])
	local gradCriterion = loss:backward(pred, Y[i])
	net:zeroGradParameters() 
	net:backward(X[i], gradLoss)
	net:updateParameters(learningRate)
end

-- Stochastic Gradient Descent (SGD) with mini-batch
-- Mini-Batch Gradient Descent 
for i = 1, N, batchSize do
	net:zeroGradParameters()
	for j = 0, batchSize-1 do
		if i+j > N then break end
		local pred = net:forward(X[i])
		local err = loss:forward(pred, Y[i])
		local gradCriterion = loss:backward(pred, Y[i])
		net:backward(X[i], gradCriterion)
	end
	net:updateParameters(learningRate)
end

--[[ torch nn module contains StochasticGradient class to do this computatin automatically. We just need to prepare the dataset according to their format described here 
http://nn.readthedocs.io/en/rtd/training/index.html#nn.StochasticGradientTrain

The code is given below. For more details on StochasticGradient class,
take a look at their code in github
http://nn.readthedocs.io/en/rtd/training/index.html#nn.StochasticGradientTrain
--]]

--[[ also take a look at the XOR training example in the documentation
http://nn.readthedocs.io/en/rtd/training/index.html#nn.traningneuralnet.dok
--]]

local dataset = {}
function dataset:size() return N end
for i = 1, N do dataset[i] = {X[i], Y[i]} end

local trainer = nn.StochasticGradient(net, loss)
trainer:train(dataset)

-- after training, test for a single test x as follows
print(net:forward(x))





