with open('dataset/Train.txt','r')as f:
    data = f.readlines()
with open('dataset/Sample.txt','w')as f:
    for line in data[:10]:
        f.write(line)
