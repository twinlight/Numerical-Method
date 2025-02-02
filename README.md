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

