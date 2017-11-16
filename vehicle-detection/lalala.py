# import matplotlib.pyplot as plt
# plt.plot([1,2,3])
# plt.savefig('myfig')


import matplotlib.image as mpimg
img = mpimg.imread("myfig.png")
mpimg.imsave("out.png", img)
