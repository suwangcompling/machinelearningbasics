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
<code>>>> lr = lin_reg.Lin_Reg('house_price.txt')<\code>
<code>>>> lr.cost()<\code>  
<code>array([ 32.07273388])<\code>  
<code>>>> lr.get_weights()<\code>  
<code>array([[ 0.],<\code>  
<code>       [ 0.]])<\code>  
<code>>>> lr.grad_desc(0.01,1500)  
<code>>>> lr.get_weights()  
<code>array([[-3.63029144],  
<code>       [ 1.16636235]])  
<code>>>> lr.cost()  
<code>array([ 4.48338826])


