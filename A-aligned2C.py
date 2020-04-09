import os, shutil

# Usage: Change file structures of your aligned data to fit homework format

src = "A-aligned"
dst = "C_224"
if not os.path.exists(dst):
    os.mkdir(dst)

for name in os.listdir(src):
    sub_src = os.path.join(src,name)
    for j in os.listdir(sub_src):
        srcpath = os.path.join(sub_src,j)
        dstpath = os.path.join(dst,j)
        shutil.move(srcpath,dstpath)
