# ml
Introductory Machine Learning in Python (for tutorial pdf visit suwangcompling.com)

# Linear Regression Usage Example
File: comma separated house_price.txt  
area  num_bedrooms  price  
2104,3,399900  
1600,3,329900  
2400,3,369000  
1416,2,232000  
3000,4,539900  
1985,4,299900  
...  
>>> lr = lin_reg.Lin_Reg('house_price.txt')  
>>> lr.cost()  
array([ 32.07273388])  
>>> lr.get_weights()  
array([[ 0.],  
       [ 0.]])  
>>> lr.grad_desc(0.01,1500)  
>>> lr.get_weights()  
array([[-3.63029144],  
       [ 1.16636235]])  
>>> lr.cost()  
array([ 4.48338826])  


