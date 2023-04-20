Newest verions of code for R2PCA (working decently well on microscopy data):

==== Core functions: ====
 - astronomy/R2PCA_astronomy.m
	This runs R2PCA based on the provided astronomy code. 
	It has 2 sub-functions defined within it, one called LoR (for first step, to recon U), the second called Sp (for second step, to recon Coeffs). 
	The last two input args (timelimit, verbose) are optional, you can exclude them when calling the function. I've been running with timelimit=false, and verbose=1. 

 - astronomy/R2PCA_astronomy_par.m
	This does the same thing, but uses parfor loop in Sp function. 
	If you want to do this, you can just run and matlab should automatically start a parallel pool if one doesnt exist. 

 - noisy/nR2PCA.m : runs the noisy variant of R2PCA. Use noisy/script_r2pca.m to test this out on simulated data (using simulation approach from R2PCA paper/code)


==== Scripts Robert used ====

 - BMC/BMC_test.m : run R2PCA_astronomy() on simulated car images from BMC2012 dataset
 	[This works nearly perfect]

 - microscopy/microscopy_main_script_crop.m : crop microscopy images to 100x100 ROI in top left corner. Run R2PCA_astronomy on these cropped images. Use rank r=2,...,5, tol=1e-3,...,1e-9

 - microscopy/microscopy_main_script.m : run R2PCA on full sized (downsampled by 2) microscopy images. Use r=2,3 tol=1e-3,1e-4

 - 







To see some microscopy results, (saved as .mat files), from cropped and full slice images, visit Google Drive folder here:
 https://drive.google.com/drive/folders/1VU9KYbNpS8OFpuPA2y41EuLwqyzw-Vap?usp=sharing


	
	