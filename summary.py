import os, glob, time
import pandas as pd

if __name__ == "__main__":
    root_dir = "./CASIA-maxpy-clean-aligned"
    files = glob.glob(root_dir+"/*/*")
    start = time.time()
    df = pd.DataFrame()
    for idx, file in enumerate(files):
        if idx%10000 == 0:print("[{}/{}]".format(idx, len(files)-1))
        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        df = df.append({'id': face_id, 'name': face_label}, ignore_index = True)
    df = df.sort_values(by = ['name', 'id']).reset_index(drop = True)
    df['class'] = pd.factorize(df['name'])[0]
    df.to_csv("summary.csv", index = False, encoding="utf-8")
    print("%.3f secs"%(time.time()-start))
