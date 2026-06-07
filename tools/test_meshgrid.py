import jittor as jt

x = jt.linspace(0, 1, 3)
y = jt.linspace(0, 1, 2)

for shape in [(2, 3), [2, 3]]:
    try:
        uu = x.reshape(1, -1).broadcast(shape)
        print("broadcast shape", shape, "->", uu.shape, uu.numpy())
    except Exception as e:
        print("broadcast shape", shape, "failed:", e)

try:
    uu = jt.broadcast(x.reshape(1, -1), (2, 3))
    print("jt.broadcast", uu.shape, uu.numpy())
except Exception as e:
    print("jt.broadcast failed:", e)

a, b = jt.meshgrid(x, y)
print("meshgrid a", a.shape, a.numpy())
print("meshgrid b", b.shape, b.numpy())

# transpose ij meshgrid to xy
print("meshgrid transpose", a.transpose().shape, a.transpose().numpy())
