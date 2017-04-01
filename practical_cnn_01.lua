--[[ install pretty-nn package first
luarocks install pretty-nn

you can run this code by typing
$ qlua <FILE_NAME> 
not 'lua' but 'qlua'

The code is taken from the tutorial
https://www.youtube.com/watch?v=BCensUz_gQ8&index=8&list=PLLHTzKZzVU9ebuL6DCclzI54MrPNFGqbW

However, the comments are written by me - Shubhra Aich (s.aich.72@gmail.com)
--]]

require 'nn'
require 'pretty-nn'

net = nn.Sequential()

-- first layer
net:add(nn.SpatialConvolution(3,6,5,5,1,1,2,2))
net:add(nn.ReLU())
--[[ 
https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution
module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
nInputPlane: #input channels 
nOutputPlane: #feature maps
kW, kH: kernel width and height
dW, dH: stride along width and height dimensions. Defaults are 1, 1.
padW, padH: padding along width and height on both sides. Defaults are 0, 0.
--]]

-- second layer
net:add(nn.SpatialConvolution(6,6,5,5,1,1,2,2))
net:add(nn.ReLU())

-- third layer
net:add(nn.SpatialConvolution(6,6,5,5,1,1,2,2))
net:add(nn.ReLU())

net:add(nn.View(-1))

-- lets define the input image
x = torch.Tensor(3, 256, 256)
-- get the size of the output of this net
print(#net:forward(x))

K = 1000 -- #classes
net:add(nn.Linear(393216, 1000))
print(net)


