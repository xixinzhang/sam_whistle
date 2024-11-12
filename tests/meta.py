import os
import json

train_files = os.listdir("/storage/DCLDE/train_puli")
test_files = os.listdir("/storage/DCLDE/test_puli")
print(train_files)
print(test_files)
train_size = [os.stat(f"/storage/DCLDE/train_puli/{file}").st_size for file in train_files]
test_size = [os.stat(f"/storage/DCLDE/test_puli/{file}").st_size for file in test_files]
print(train_size)
print(test_size)


use_train = [f"/storage/DCLDE/train_puli/{file}" for file in train_files if os.stat(f"/storage/DCLDE/train_puli/{file}").st_size > 0]
use_test = [f"/storage/DCLDE/test_puli/{file}" for file in test_files if os.stat(f"/storage/DCLDE/test_puli/{file}").st_size > 0]
print(use_train, len(use_train))
print(use_test, len(use_test))

train_stems = [file.split(".")[0] for file in use_train]
test_stems = [file.split(".")[0] for file in use_test]

# data_meta = {"train": train_stems, "test": test_stems}
# json.dump(data_meta, open("data/dclde/meta.json", "w"), indent=4)
for file in use_train+use_test:
    os.symlink(file, os.path.join("data/dclde/annotation", os.path.basename(file)))
    
