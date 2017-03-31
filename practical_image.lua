-- you can run this code by typing
-- $ qlua <FILE_NAME> 
-- not 'lua' but 'qlua'

require 'image'
im = image.load('lena.jpg')
print(#im) -- = #im (OR) return #im for command line

print(torch.type(im)) -- = torch.type(im) for cl
--image.display(im)
g = im:clone() -- deep clone
print(#g) -- = #g for cl
g[1]:fill(0) -- red plane
g[3]:fill(0) -- blue plane
--image.display(g)
--image.display(image={im,g}, legend='orignal | green')

r = torch.zeros(#im)
r[1]=im[1]
b = torch.zeros(#im)
b[3] = im[3]
-- image.display{image={im,r,g,b}, nrow=2, legend='orignal|red|green|blue'}
-- another way to display combined image
im_out = image.toDisplayTensor{input={im,r,g,b}, nrow=2, legend='orignal|red|green|blue'}
image.display{image=im_out, legend='output image'}
image.saveJPG('./lena-four.jpg', im_out) -- has problem with savePNG

im_crop = image.crop(im_out, 0, 0, 300, 400) -- crop image
image.display(im_crop)


