def gen(streams):
    for s in streams:
        s.send(None)
    for i in range(10):
        s = streams[i%2]
        s.send(i)

def stream():
    while True:
        i = yield
        yield i

streams = [stream(), stream()]

print(next(gen(streams)))

