import pack1.pack_file_1

from pack1 import pack1_var_1, pack1_var_2, pack1_var_3

if __name__ == "__main__":
    print(f"this is {__name__} and I import pack1")

    print(pack1_var_1)
    print(pack1_var_2)
    print(pack1_var_3)
    print(pack1.pack_file_1.pack1_func_1())
    print(pack1.pack_file_1.pack1_func_2())