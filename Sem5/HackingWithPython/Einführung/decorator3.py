import time

def addition(x,y):
    print("Addition:")
    return x+y

def subtraction(x,y):
    print("Subtraction:")
    return x-y

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

theAdd1 = decorate_time(addition)
theAdd2 = decorate_print(theAdd1)

theSub1 = decorate_time(subtraction)
theSub2 = decorate_print(theSub1)

print(theAdd2(3,3))
print(theSub2(4,3))