def myfunc(*args, **kwargs):
    print(len(args))
    print(len(kwargs))

    for i in args:
        print(i)

    for k, v, in kwargs.items():
        print(f"{k} : {v}")

myfunc(2, 3, first = "my first", second = 3)