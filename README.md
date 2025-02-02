Lab1: Bisection method


f = input('Enter the function: ', 's');
f = inline(f);
E = 0.00005;

a = input('Enter the start root interval: ');
b = input('Enter the end root interval: ');

fa = f(a);
fb = f(b);


while(fa*fb > 0)

  a = input('Enter the start root interval: ');
  b = input('Enter the end root interval: ');


  fa = f(a);
  fb = f(b);
end
step = 0;
  disp('Step a         f(a)         b           f(b)        x        f(x)');

while(abs(b-a)>E)

  x = (a+b)/2;
  fx=f(x);
  step =step + 1;
  result = [step,a,f(a),b,f(b),x,f(x)];
  disp(result);
  
  if(f(x)==0)
  result = strcat('The root lies at x = ',num2str(x));
  disp(result); 
  end
 
 if(f(a)*f(x)<0)
  b=x;
 else
  a=x;
 end
 

end


if (f(x) ~= 0)
  disp('-----------------------------');

  result = strcat('The root lies at x = ', num2str(x));
  disp(result);
end







Lab 2: Newton Ramphson method





def f(x):
    return x**3 - 5*x - 9

def g(x):
    return 3*x**2 - 5


def newtonRaphson(x0,e,N):
    print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        if g(x0) == 0.0:
            print('Divide by zero error!')
            break
        
        x1 = x0 - f(x0)/g(x0)
        print('Iteration-%d, x1 = %0.6f and f(x1) = %0.6f' % (step, x1, f(x1)))
        x0 = x1
        step = step + 1
        
        if step > N:
            flag = 0
            break
        
        condition = abs(f(x1)) > e
    
    if flag==1:
        print('\nRequired root is: %0.8f' % x1)
    else:
        print('\nNot Convergent.')



x0 = input('Enter Guess: ')
e = input('Tolerable Error: ')
N = input('Maximum Step: ')

x0 = float(x0)
e = float(e)
N = int(N)

newtonRaphson(x0,e,N)






Lab 2: Secant Method






def f(x):
    return x**3 - 5*x - 9


def secant(x0,x1,e,N):
    print('\n\n*** SECANT METHOD IMPLEMENTATION ***')
    step = 1
    condition = True
    while condition:
        if f(x0) == f(x1):
            print('Divide by zero error!')
            break
        
        x2 = x0 - (x1-x0)*f(x0)/( f(x1) - f(x0) ) 
        print('Iteration-%d, x2 = %0.6f and f(x2) = %0.6f' % (step, x2, f(x2)))
        x0 = x1
        x1 = x2
        step = step + 1
        
        if step > N:
            print('Not Convergent!')
            break
        
        condition = abs(f(x2)) > e
    print('\n Required root is: %0.8f' % x2)



x0 = input('Enter First Guess: ')
x1 = input('Enter Second Guess: ')
e = input('Tolerable Error: ')
N = input('Maximum Step: ')


x0 = float(x0)
x1 = float(x1)
e = float(e)
N = int(N)

secant(x0,x1,e,N)








Lab 3 : Lagrange Interpolation





import numpy as np


n = int(input('Enter number of data points: '))


x = np.zeros((n))
y = np.zeros((n))



print('Enter data for x and y: ')
for i in range(n):
    x[i] = float(input( 'x['+str(i)+']='))
    y[i] = float(input( 'y['+str(i)+']='))



xp = float(input('Enter interpolation point: '))


yp = 0

for i in range(n):
    
    p = 1
    
    for j in range(n):
        if i != j:
            p = p * (xp - x[j])/(x[i] - x[j])
    
    yp = yp + p * y[i]    

print('Interpolated value at %.3f is %.3f.' % (xp, yp))






Lab 3: Newton's Interpolation





def u_cal(u, n):

	temp = u;
	for i in range(1, n):
		temp = temp * (u - i);
	return temp;

def fact(n):
	f = 1;
	for i in range(2, n + 1):
		f *= i;
	return f;

n = 4;
x = [ 45, 50, 55, 60 ];
	

y = [[0 for i in range(n)]
		for j in range(n)];
y[0][0] = 0.7071;
y[1][0] = 0.7660;
y[2][0] = 0.8192;
y[3][0] = 0.8660;


for i in range(1, n):
	for j in range(n - i):
		y[j][i] = y[j + 1][i - 1] - y[j][i - 1];

for i in range(n):
	print(x[i], end = "\t");
	for j in range(n - i):
		print(y[i][j], end = "\t");
	print("");

value = 52;


sum = y[0][0];
u = (value - x[0]) / (x[1] - x[0]);
for i in range(1,n):
	sum = sum + (u_cal(u, i) * y[0][i]) / fact(i);

print("\nValue at", value, 
	"is", round(sum, 6));





Lab 4: 

///////Trapezodial rule///////






def f(x):
    return 1/(1 + x**2)


def trapezoidal(x0,xn,n):
   
    h = (xn - x0) / n
    
   
    integration = f(x0) + f(xn)
    
    for i in range(1,n):
        k = x0 + i*h
        integration = integration + 2 * f(k)
    
    
    integration = integration * h/2
    
    return integration
    

lower_limit = float(input("Enter lower limit of integration: "))
upper_limit = float(input("Enter upper limit of integration: "))
sub_interval = int(input("Enter number of sub intervals: "))

result = trapezoidal(lower_limit, upper_limit, sub_interval)
print("Integration result by Trapezoidal method is: %0.6f" % (result) )







///// Simpsons 1/3 rule ///////






def f(x):
    return 1/(1 + x**2)

def simpson13(x0,xn,n):
    
    h = (xn - x0) / n
    
    
    integration = f(x0) + f(xn)
    
    for i in range(1,n):
        k = x0 + i*h
        
        if i%2 == 0:
            integration = integration + 2 * f(k)
        else:
            integration = integration + 4 * f(k)
    
    
    integration = integration * h/3
    
    return integration
    

lower_limit = float(input("Enter lower limit of integration: "))
upper_limit = float(input("Enter upper limit of integration: "))
sub_interval = int(input("Enter number of sub intervals: "))

result = simpson13(lower_limit, upper_limit, sub_interval)
print("Integration result by Simpson's 1/3 method is: %0.6f" % (result) )







///// Simpsons 3/8 rule ////////




def f(x):
    return 1/(1 + x**2)


def simpson38(x0,xn,n):
    # calculating step size
    h = (xn - x0) / n
    
  
    integration = f(x0) + f(xn)
    
    for i in range(1,n):
        k = x0 + i*h
        
        if i%3 == 0:
            integration = integration + 2 * f(k)
        else:
            integration = integration + 3 * f(k)
    
    
    integration = integration * 3 * h / 8
    
    return integration
    

lower_limit = float(input("Enter lower limit of integration: "))
upper_limit = float(input("Enter upper limit of integration: "))
sub_interval = int(input("Enter number of sub intervals: "))

result = simpson38(lower_limit, upper_limit, sub_interval)
print("Integration result by Simpson's 3/8 method is: %0.6f" % (result) )






Lab 5: Romberg Integration



import numpy as np

def romberg_integration(f, a, b, tol=1e-6, max_iter=20):

    R = np.zeros((max_iter, max_iter), dtype=float)

   
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    
    for i in range(1, max_iter):
        h /= 2  
        
        sum_f = 0
        for k in range(1, 2**i, 2):
            sum_f += f(a + k * h)
        R[i, 0] = 0.5 * R[i-1, 0] + h * sum_f

        
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

   
        if abs(R[i, i] - R[i-1, i-1]) < tol:
            return R[i, i], i + 1

    
    print("Maximum iterations reached. Result may not be fully converged.")
    return R[-1, -1], max_iter


if __name__ == "__main__":
  
    f = lambda x: np.exp(-x**2)  

  
    a = 0
    b = 1

  
    integral, iterations = romberg_integration(f, a, b)

    print(f"Approximated integral: {integral:.8f}")
    print(f"Iterations performed: {iterations}")







  /////// Runge Kutta 4th order ////////////





  def f(x,y):
    return x+y


def rk4(x0,y0,xn,n):
    
  
    h = (xn-x0)/n
    
    print('\n--------SOLUTION--------')
    print('-------------------------')    
    print('x0\ty0\tyn')
    print('-------------------------')
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k
        print('%.4f\t%.4f\t%.4f'% (x0,y0,yn) )
        print('-------------------------')
        y0 = yn
        x0 = x0+h
    
    print('\nAt x=%.4f, y=%.4f' %(xn,yn))


print('Enter initial conditions:')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('Enter calculation point: ')
xn = float(input('xn = '))

print('Enter number of steps:')
step = int(input('Number of steps = '))

rk4(x0,y0,xn,step)








////// Rungee kutta  1st order //////






import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])  
        x[i + 1] = x[i] + h

    return x, y


if __name__ == "__main__":
 
    f = lambda x, y: x + y  


    x0 = 0
    y0 = 1

   
    h = 0.1
    n = 10

   
    x_euler, y_euler = euler_method(f, x0, y0, h, n)

   
    print("1st-order Runge-Kutta (Euler's method):")
    print("x:", x_euler)
    print("y:", y_euler)

   
    plt.plot(x_euler, y_euler, label="Euler's method", marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1st-order Runge-Kutta (Euler's method)")
    plt.legend()
    plt.grid()
    plt.show()







  /////// Runge kutta 2nd order ////////



  import numpy as np
import matplotlib.pyplot as plt

def midpoint_method(f, x0, y0, h, n):
  
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0

    for i in range(n):
        k1 = h * f(x[i], y[i])  
        k2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k1) 
        y[i + 1] = y[i] + k2  
        x[i + 1] = x[i] + h

    return x, y


if __name__ == "__main__":
 
    f = lambda x, y: x + y  

   
    x0 = 0
    y0 = 1

    h = 0.1
    n = 10

    
    x_midpoint, y_midpoint = midpoint_method(f, x0, y0, h, n)

    
    print("2nd-order Runge-Kutta (Midpoint method):")
    print("x:", x_midpoint)
    print("y:", y_midpoint)

    plt.plot(x_midpoint, y_midpoint, label="Midpoint method", marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2nd-order Runge-Kutta (Midpoint method)")
    plt.legend()
    plt.grid()
    plt.show()









