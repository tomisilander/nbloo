f = open('iris.idt')
rows = (l.split() for l in f)
cols = zip(*rows)
cs = map(Counter, cols)
# list(cs)

