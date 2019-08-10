import numpy as np
A = np.array([[255,   0, 255,   0,   0],
   [  0, 255,   0,   0,   0],
   [  0,   0, 255,   0, 255],
   [  0, 255, 255, 255, 255],
   [255,   0, 255,   0, 255]])

B = np.array([[255,   0, 255,   0, 255],
   [  0, 255,   0,   0,   0],
   [255,   0,   0,   0, 255],
   [  0,   0, 255, 255, 255],
   [255,   0, 255,   0,   0]])

number_of_equal_elements = np.sum(A==B)
total_elements = np.multiply(*A.shape)
percentage = number_of_equal_elements/total_elements

print('total number of elements: \t\t{}'.format(total_elements))
print('number of identical elements: \t\t{}'.format(number_of_equal_elements))
print('number of different elements: \t\t{}'.format(total_elements-number_of_equal_elements))
print('percentage of identical elements: \t{:.2f}%'.format(percentage*100))