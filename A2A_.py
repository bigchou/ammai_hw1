import shutil, os

# Usage: Change file structures of your raw data to fit alignment tool requirements

src = "A"
dst = "A_"
if not os.path.exists(dst):
    os.mkdir(dst)

for i in os.listdir(src):
    name = i.split("_")[0]
    if not os.path.exists(os.path.join(dst,name)):
        os.mkdir(os.path.join(dst,name))
    srcpath = os.path.join(src,i)
    dstpath = os.path.join(dst,name,i)
    shutil.move(srcpath,dstpath)
