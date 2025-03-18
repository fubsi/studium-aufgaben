import time

def decorate_print(func):
    
    def inner(*args, **kwargs):
        print("->decorate_print()")
        val = func(*args, **kwargs)
        print("<-decorate_print()")
        return val
    
    return inner

def decorate_time(func):

    def inner(*args, **kwargs):
        print("->decorate_time()")
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print(f"<-decorate_time(): {end-start}")
        return val
    
    return inner

@decorate_print
@decorate_time
def addition(x,y):
    print("Addition:")
    return x+y

@decorate_time
@decorate_print
def subtraction(x,y):
    print("Subtraction:")
    return x-y

print(addition(3,3))
print(subtraction(4,3))