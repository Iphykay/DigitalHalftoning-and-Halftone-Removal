import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random

# Setting the values
random.seed(10)

#Loading one of the image
image_DHT = "C:/Users/Starboy/OneDrive - rit.edu/Courses/Grad Lab/Fourier Project/Images/Image1.jpg"
image_HT = "C:/Users/Starboy/OneDrive - rit.edu/Grad Lab/Images/Kahanamoku_Jan-37_cropped_bitmap.tif"


class DigitalHalftoning:
    """
    This class is used to convert photographs to bitonal images ie dots of black ink of white
    paper. The class also contains two process that helps in achieving this.
    Independent Quantization and Error diffused Error Quantization.
    """

    def __init__(self, image):
        self.image = image


    def readimage(self, image):
        """
        This method read the file and converts it to gray-scale.
        """

        image = self.image
        read_image = cv2.imread(image)
        image_gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)

        return image_gray
    
    def displayimage(self, image):
        """
        This method helps in diplaying the image.
        """
        if isinstance(image, (float, int)):
            self.image = image
        output = plt.imshow(image, cmap='gray')
        output = plt.show()
        return output

    def normalizeimages(self, image):
        self.image = image
        return self.image/255.0

    def normalization(self, image):
        """
        This helps in normalizing the real sides of the DFT
        """

        self.image = image
        max_real, min_real = np.max(self.image.real), np.min(self.image.real)
        result = (self.image.real - min_real)/(max_real - min_real)
        return result

    def independent_quantization(self, image):
        """
        This method employs the Independent Quantization 
        which take an input image and thresholds it using the step function.
        """

        self.image = image
        m, n = np.max(self.image), np.min(self.image)
        x0 = 0.5*(m-n)
        r = np.array(np.zeros(shape=(self.image.shape[0], self.image.shape[1])))
        r[np.where(self.image > x0)] = 1.0
        r[np.where(self.image < x0)] = 0.0
        return r

    def error_diffused_quantization(self, image):
        """
        This method uses the error diffused quantization by Floyd and Steinberg
        to reduce the quantization error at subsequent pixels.
        """
        
        self.image = image
        scaled_image = np.pad(self.image / 255, 1) # pad the image to handle the borders
        Image_output = np.zeros(scaled_image.shape)
        row, col = Image_output.shape

        floyd_steinberg_weights = np.array([7/16, 1/16, 5/16, 3/16])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                Image_output[i, j] = int(scaled_image[i, j] + 0.5) 
                error = scaled_image[i, j] - Image_output[i, j]
                scaled_image[((i , i+1, i+1, i+1),(j+1, j+1, j , j-1 ))] += error*floyd_steinberg_weights
        Image_output = Image_output[1:row-1, 1:col -1]
        return Image_output

    def dft(self, image):
        """
        This calculates the dft of an image.
        """
        self.image = image
        Image_1_dft = ((1/np.sqrt(self.image.shape[0]-1))*(1/np.sqrt(self.image.shape[1]-1)))*np.fft.fft2(self.image)
        Image_1_dft_shift = np.fft.fftshift(Image_1_dft) 
        return np.log(np.abs(Image_1_dft_shift))

    def inverse_dff(self, image):
        """
        This calculates the inverse dft of an image.
        """
        self.image = image
        InverseDFT = np.fft.ifftshift(self.image)
        InverseDFT = np.fft.ifft2(InverseDFT)
        return InverseDFT

    @staticmethod
    def magnitude(x):
        return np.log(np.abs(x))

    def histogram(self, image, return_values=False):

        self.image = image
        row, col = self.image.shape
        value, count = np.unique(self.image.ravel(), return_counts=True)
        norm_count = count/(row*col)
        if return_values == False:
            plt.plot(value, norm_count,'k.-')
            plt.show()
        else:

            return value, norm_count
    
    def gaussian_filter(self, sigma, image):
        self.image = image
        row, col = self.image.shape
        yrow, ycol = int(row/2),int(col/2)
        X, Y = np.linspace(-ycol,ycol,col), np.linspace(yrow,-yrow,row)
        x, y = np.meshgrid(X, Y)
        normal = 1 / (2.0*np.pi*sigma**2)
        Gauss =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return Gauss
    
    def plot_for_quantization_errors(self, value, norm_count, values, Histo_shift_F, name: str):
        fig, axes = plt.subplots(ncols=2, figsize=(15,5))
        axes[0].plot(value, norm_count,'k-')
        axes[0].set_xlabel('Error')
        axes[0].set_title('Normalized histogram: Quantization Error for Image')
        axes[1].plot(values, Histo_shift_F,'r-')
        axes[1].set_xlabel('k')
        axes[1].set_title(name +': Error Spectra for Image')
        plt.show() 

    def error_independent_quantized(self, quant_image, norm_images, name:str):
        min_norm_image, max_norm_image = np.min(norm_images), np.max(norm_images)
        Error = (quant_image*(max_norm_image-min_norm_image)+min_norm_image)-norm_images
        value, count = self.histogram(Error, return_values=True)
        norm_count = count/sum(count)

        Histo_Fourier = np.fft.fft(count)
        Histo_shift_F = np.fft.fftshift(np.abs(Histo_Fourier))
    
        Length = Histo_shift_F.shape
        Histo_shift_F[int(Length[0]/2)] = 0
        values = np.linspace(-int(Length[0]/2),int(Length[0]/2)+1,Length[0])
        return self.plot_for_quantization_errors(value, norm_count, values, Histo_shift_F, name)
       
    def fourier_error_diffuse_quantized(self, quant_image, norm_images, name: str):

        """
        This method plots the histogram of the quantization errors and evaluate 
        their spectra vias DFT for the error diffused quantization
        """

        Error_images = quant_image - norm_images

        #Plot the histogram
        value, count = self.histogram(Error_images, return_values=True)
        norm_count = count/sum(count)
        Histo_Fourier = np.fft.fft(count)
        Histo_shift_F = np.fft.fftshift(np.abs(Histo_Fourier))

        #Deleting the central ordinate
        Length = Histo_shift_F.shape
        Histo_shift_F[int(Length[0]/2)] = 0
        values = np.linspace(-int(Length[0]/2),int(Length[0]/2)+1,Length[0])

        return self.plot_for_quantization_errors(value, norm_count, values, Histo_shift_F, name)

    def noise_creation(self, image, name:str):

        """
        This method creates white and blue noise.
        """

        self.image = image
        phase_white = np.random.uniform(-np.pi,np.pi,size=(self.image.shape[0],self.image.shape[1]))
        real_part = np.cos(phase_white); imaginary_part = np.sin(phase_white)
        noise = np.vectorize(complex)(real_part, imaginary_part)
        Blue_noise = 1.0 - self.gaussian_filter(100, image)
        if name == 'White Noise':
            noise = noise
        elif name == 'Blue Noise':
            noise = Blue_noise * noise
        
        Noise = self.inverse_dff(noise)

        #Normalizing the noise
        Noise = self.normalization(Noise)

        #Add image with Noise
        Noise += self.normalizeimages(image)

        #Thresholding
        Noise = self.independent_quantization(Noise)

        #Display Output
        return self.displayimage(Noise)
    
    def run_digitalhalftoning(self):
        DHTT = self.readimage(image_DHT)
        self.displayimage(DHTT)
        self.normalizeimages(DHTT)
        self.independent_quantization(DHTT)
        self.error_diffused_quantization(DHTT)
        self.displayimage(self.dft(DHTT))
        # print(DHT.noise_creation(DHTT, 'White Noise'))
        # print(DHT.noise_creation(DHTT, 'Blue Noise'))
        self.displayimage(self.independent_quantization(DHTT))
        self.displayimage(self.error_diffused_quantization(DHTT))
        self.error_independent_quantized(self.independent_quantization(DHTT),self.normalizeimages(DHTT),'Independent Error Quantization')
        self.fourier_error_diffuse_quantized(self.error_diffused_quantization(DHTT),self.normalizeimages(DHTT), 'Error-Diffused Quantization')
        
    
    


class HalftoningRemoval(DigitalHalftoning):
    """
    This class helps in removing halftone images via DFT by generating an approximation 
    of a gray scale image from a bitoned halftoned input.
    """
    def __init__(self, image, name='Halftone Removal'):
        super().__init__(image)


    def readimage(self, image):
        return super().readimage(image)
    

    def displayimage(self, image):
        return super().displayimage(image)
    
    
    def dft(self, image):
        return super().dft(image)
    

    def inverse_dff(self, image):
        Shift_dft = np.fft.fftshift(image)
        return super().inverse_dff(Shift_dft)
    
    
    def rectangular_filter(self, image):
        self.image = image
        row, col = self.image.shape
        yrow, ycol = int(row/2),int(col/2)

        X, Y = np.linspace(-ycol,ycol,col), np.linspace(yrow,-yrow,row)

        Rect_Filter = np.zeros(shape=(row,col))
        x, y = np.meshgrid(X,Y)
        Rect_Filter[np.where(abs(x) <= 0.5*250)] = 1.0
        Rect_Filter[np.where(abs(y) <= 0.5*250)] = 1.0
        Rect_Filter[np.where(abs(x) > 0.5*250)] = 0.0
        Rect_Filter[np.where(abs(y) > 0.5*250)] = 0.0

        return Rect_Filter
    
    
    def cyclic_filter(self, image):
        self.image = image
        row, col = self.image.shape
        yrow, ycol = int(row/2),int(col/2)

        mask = np.ones((row, col), np.float32)
        r = 100
        center = [yrow, ycol]
        x, y = np.ogrid[:row, :col]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
        mask[mask_area] = 0

        return mask
    
    def run_halftoningremoval(self):
        HTI = self.readimage(image_HT)
        self.displayimage(HTI)
        self.displayimage(self.dft(HTI))
        self.displayimage(self.rectangular_filter(HTI))


def main():
    simulate = DigitalHalftoning(image_DHT)
    simulate.run_digitalhalftoning()

def main2():
    simulate = HalftoningRemoval(image_HT)
    simulate.run_halftoningremoval()

if __name__ == "__main__":
    main()
    main2()






