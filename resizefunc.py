from PIL import Image
import numpy as np
import  cv2


target = np.random.rand(512,512)
sum1 = target.sum()

# target = np.asarray(img)

# print(target.shape)
tmp= cv2.resize(target, (int(target.shape[0] * 8), int(target.shape[1] * 8)), interpolation=cv2.INTER_NEAREST)

tmp1 = np.array(tmp)

sum2 = tmp1.sum()

times = sum2/sum1

print('sum1 : {} , sum2 :{}, times: {}'.format(sum1, sum2, times))

j = 1


