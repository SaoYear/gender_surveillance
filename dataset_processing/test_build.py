import os

file_dir = 'DIR'
target_dir = 'DIR'
filename = []
for _, _, _file in os.walk(file_dir):
    filename = _file
print(filename)

for _id, file in enumerate(filename):
    if _id % 5 == 0:
        os.rename(file_dir + file, target_dir + file)
