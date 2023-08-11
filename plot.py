import h5py
import matplotlib.pyplot as plt

f = h5py.File('../result/test_Unet/reconstructions_val/brain_acc4_179.h5', 'r')
# f = h5py.File('YOUR FILE PATH', 'r')

input = f['input']
recon = f['reconstruction']
target = f['target']

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(input[1, 0, :, :])
plt.title('input')
plt.subplot(2, 2, 2)
plt.imshow(input[1, 1, :, :])
plt.title('grappa')
plt.subplot(2, 2, 3)
plt.imshow(recon[1, :, :])
plt.title('reconstruction')
plt.subplot(2, 2, 4)
plt.imshow(target[1, :, :])
plt.title('target')
plt.savefig('result.png', dpi=300)
