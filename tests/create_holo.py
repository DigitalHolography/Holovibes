import sys, os

os.system("clang.exe create_holo.cc")
cmd = "a.exe "
for arg in sys.argv[1:]:
    cmd += arg + " "
os.system(cmd)
os.system("rm a.exe")