import numpy as np
import jittor as jt

y = jt.array(np.array([-1.0, 0.0, 2.0], dtype=np.float32))
jt.sync_all()
print("y", y.numpy())
print("jt.expm1", jt.expm1(y.abs()).numpy())
sign = jt.where(y > 0, 1.0, jt.where(y < 0, -1.0, 0.0))
jt.sync_all()
print("where sign", sign.numpy())
print("inv_log", (sign * jt.expm1(y.abs())).numpy())
