# denoising
Contains scripts written for multi-purpose denoising tasks. 

Files: Universal_Encoder_Decoder.m : Uses a third-party basis pursuit algorithm to recover sparse signals. 

Split_Bregman_Anisotropic_FixedBCs.m : A stand-alone script for denoising.  Algorithms are from "The Split Bregman Method for L1 Regularized Prroblems" by Osher and Goldstein. 

ROF_SplitBreg.m + SplitBregmanROF.(all file extensions) : A multigrid denoising method based off of algorithms originally written by Tom Goldstein, UMD CS department.  The multigrid modification and plotting utilities are new.  The multigrid denoising aspect is entirely unstudied, it was just an idea I came up with to solve a problem at hand. 
