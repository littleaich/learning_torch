-- this is my first code

local M = {}

local function sayMyName()
	print('Aich')
end

function M.sayHello()
	print ('Hello')
	sayMyName()
end

print('all code loaded')

return M

