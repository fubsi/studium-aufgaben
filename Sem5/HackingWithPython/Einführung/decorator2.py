def addition(x,y):
    print("Addition:")
    return x+y

def subtraction(x,y):
    print("Subtraction:")
    return x-y

def func(x,y,func):
    return func(x,y)

print(func(2,3,addition))
print(func(2,3,subtraction))