import matplotlib.pyplot as plt
from Variational_AutoEncoder.models.dataset_transform import ScatteringTransform
# Instantiate your ScatteringTransform model
model = ScatteringTransform(input_size=2400, input_dim=1, log_stat=(np.zeros(13), np.ones(13)), device='cpu')

# Extract the filter banks from the ScatteringNet
phi, psi = model.transform.phi, model.transform.psi

# Plotting the low-pass filter (phi)
plt.figure()
plt.plot(np.fft.fftshift(np.abs(phi)))
plt.title('Low-pass filter')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

# Plotting the wavelet filters (psi)
for j in range(len(psi)):
    plt.figure()
    plt.plot(np.fft.fftshift(np.abs(psi[j])))
    plt.title(f'Wavelet filter at scale {j}')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()
