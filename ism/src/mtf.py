from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO

        eps = 1e-6

        fstepAlt = 1 / nlines / w   # 333.3333333
        fstepAct = 1 / ncolumns / w # 222.2222222

        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fAltxx, fActxx] = np.meshgrid(fAlt, fAct, indexing='ij') # Please use ‘ij’ indexing or you will get the transpose
        f2D = np.sqrt(fAltxx * fAltxx + fActxx * fActxx)

        fc = D /(lambd*focal)

        fnAlt = fAlt / (1 / w)
        fnAct = fAct / (1/w)
        fn2D = f2D / (1 / w)
        fr2D = f2D / fc

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        #TODO
        fr_clip = np.clip(fr2D, 0.0, 1.0)
        Hdiff = (2.0 / np.pi) * (
                np.arccos(fr_clip)
                - fr_clip * np.sqrt(1.0 - fr_clip ** 2)
        )

        Hdiff = np.where(fr2D <= 1.0, Hdiff, 0.0)

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        fr2D = np.asarray(fr2D, dtype=float)

        x = np.pi * defocus * fr2D * (1.0 - fr2D)

        Hdefoc = np.ones_like(x)
        mask = np.abs(x) > 1e-12
        Hdefoc[mask] = 2.0 * j1(x[mask]) / x[mask]

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        fr2D = np.asarray(fr2D, dtype=float)

        Hwfe = np.exp(
            -fr2D * (1.0 - fr2D) * (
                    kLF * (wLF / lambd) ** 2
                    + kHF * (wHF / lambd) ** 2
            )
        )

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        fn2D = np.asarray(fn2D, dtype=float)

        Hdet = np.abs(np.sinc(fn2D))

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """

        fnAlt = np.asarray(fnAlt, dtype=float)

        Hsmear = np.sinc(ksmear * fnAlt)

        Hsmear = np.tile(Hsmear[:, np.newaxis], (1, ncolumns))

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """

        fn2D = np.asarray(fn2D, dtype=float)

        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """

        os.makedirs(directory, exist_ok=True)

        centerAlt = np.argmin(np.abs(fnAlt))
        centerAct = np.argmin(np.abs(fnAct))

        positiveAct = fnAct >= 0
        positiveAlt = fnAlt >= 0

        plt.figure(figsize=(12, 6))

        plt.plot(fnAct[positiveAct], Hdiff[centerAlt, positiveAct],
                 label="Diffraction MTF")
        plt.plot(fnAct[positiveAct], Hdefoc[centerAlt, positiveAct],
                 label="Defocus MTF")
        plt.plot(fnAct[positiveAct], Hwfe[centerAlt, positiveAct],
                 label="WFE Aberrations MTF")
        plt.plot(fnAct[positiveAct], Hdet[centerAlt, positiveAct],
                 label="Detector MTF")
        plt.plot(fnAct[positiveAct], Hsmear[centerAlt, positiveAct],
                 label="Smearing MTF")
        plt.plot(fnAct[positiveAct], Hmotion[centerAlt, positiveAct],
                 label="Motion blur MTF")
        plt.plot(fnAct[positiveAct], Hsys[centerAlt, positiveAct],
                 color="black", linewidth=2, label="System MTF")

        plt.axvline(0.5, color="black", linestyle="--",
                    linewidth=2, label="f Nyquist")

        plt.title("System MTF - slice ACT")
        plt.xlabel("Spatial frequencies f/(1/w) [-]")
        plt.ylabel("MTF")
        plt.xlim(-0.02, 0.525)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.5)
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"mtf_{band}_ACT.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(12, 6))

        plt.plot(fnAlt[positiveAlt], Hdiff[positiveAlt, centerAct],
                 label="Diffraction MTF")
        plt.plot(fnAlt[positiveAlt], Hdefoc[positiveAlt, centerAct],
                 label="Defocus MTF")
        plt.plot(fnAlt[positiveAlt], Hwfe[positiveAlt, centerAct],
                 label="WFE Aberrations MTF")
        plt.plot(fnAlt[positiveAlt], Hdet[positiveAlt, centerAct],
                 label="Detector MTF")
        plt.plot(fnAlt[positiveAlt], Hsmear[positiveAlt, centerAct],
                 label="Smearing MTF")
        plt.plot(fnAlt[positiveAlt], Hmotion[positiveAlt, centerAct],
                 label="Motion blur MTF")
        plt.plot(fnAlt[positiveAlt], Hsys[positiveAlt, centerAct],
                 color="black", linewidth=2, label="System MTF")

        plt.axvline(0.5, color="black", linestyle="--",
                    linewidth=2, label="f Nyquist")

        plt.title("System MTF - slice ALT")
        plt.xlabel("Spatial frequencies f/(1/w) [-]")
        plt.ylabel("MTF")
        plt.xlim(-0.02, 0.525)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.5)
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"mtf_{band}_ALT.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))

        plt.imshow(
            Hsys,
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=np.min(Hsys),
            vmax=1
        )

        plt.xlabel("ACT")
        plt.ylabel("ALT")
        plt.title(f"System MTF for {band}")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"mtf_{band}_2D.png"), dpi=150)
        plt.close()
