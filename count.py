import os

path4K = "/home/yixiaow/projects/def-panos/SharedProject4Kto8K/4K"
path8K = "/home/yixiaow/projects/def-panos/SharedProject4Kto8K/8K"

ct4K = []
ct8K = []
for d in sorted(os.listdir(path4K)):
    ct4K.append(len(os.listdir(os.path.join(path4K, d))))
for d in sorted(os.listdir(path8K)):
    ct8K.append(len(os.listdir(os.path.join(path8K, d))))

print(ct4K)
print(ct8K)