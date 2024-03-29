from ...backend.numpy_backend import NumpyBackend


class NumpyBackend1D(NumpyBackend):
    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling in the Fourier domain
        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.
        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.
        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)

        y = x.reshape(-1, k, x.shape[-1] // k)

        res = y.mean(axis=-2)

        return res

    @classmethod
    def pad(cls, x, pad_left, pad_right):
        """Pad real 1D tensors
        1D implementation of the padding function for real PyTorch tensors.
        Parameters
        ----------
        x : tensor
            Three-dimensional input tensor with the third axis being the one to
            be padded.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        Returns
        -------
        output : tensor
            The tensor passed along the third dimension.
        """
        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')

        paddings = ((0, 0),) * len(x.shape[:-1])
        paddings += (pad_left, pad_right),

        output = cls._np.pad(x, paddings, mode='reflect')

        return output

    @staticmethod
    def unpad(x, i0, i1):
        """Unpad real 1D tensor
        Slices the input tensor at indices between i0 and i1 along the last axis.
        Parameters
        ----------
        x : tensor
            Input tensor with least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[..., i0:i1]

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)

        return cls._np.fft.fft(x)

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x).real

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x)
       
    @classmethod
    def fft(cls, x):

        return cls._np.fft.fft(x)

    @classmethod
    def cast_complex(cls, x):
        cls.real_check(x)

        return x.astype(complex)
      
    @classmethod
    def pow(cls, x, power):

        return cls._np.power(x, power)
      
    @classmethod
    def conj(cls, x):

        return cls._np.conj(x)

    @classmethod
    def real(cls, x):

        return cls._np.real(x)
    
    @classmethod
    def divide(cls, x, y):

        return cls._np.divide(x, y)

    @classmethod
    def multiply(cls, x, y):

        return cls._np.multiply(x, y)

    @classmethod
    def modulo(cls, x, y):

        return cls._np.mod(x, y)


    @classmethod
    def reshape(cls, x, new_shape):

        return cls._np.reshape(x, new_shape)
     
    @classmethod
    def to_polar(cls, x):
        cls.complex_check(x)

        r = cls.modulus(x)
        phi = cls._np.angle(x)
        return r, phi
       
    @classmethod
    def to_cartesian(cls, mag, phase):
        # temp!!!
        #cls.real_check(mag)
        cls.real_check(phase)
        
        real = cls._np.cos(phase)
        real = cls._np.multiply(real, mag)
        imag = cls._np.sin(phase)
        imag = cls._np.multiply(imag, mag)
        return cls._np.complex(real, imag)

    @classmethod
    def rms_diff(cls, x, y):
        assert cls._np.allclose(x, y)


backend = NumpyBackend1D
