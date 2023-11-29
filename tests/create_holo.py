import sys, os

os.system("clang.exe create_holo.cc")
print(sys.argv)
os.system("a.exe " + sys.argv[1])
os.system("rm a.exe")