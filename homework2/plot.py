import matplotlib.pyplot as plot
X = []
Y = []
tmp = ""
with open('log.txt') as f:
    tmp = f.readlines()

for l in tmp:
    if 'iter' in l:
        t = l.split(' ')
        X.append(int(t[1].replace(':', '')))
        Y.append(float(t[2]))

plot.plot(X,Y)
plot.plot([0,40000],[10,10], color='r')
plot.show()
