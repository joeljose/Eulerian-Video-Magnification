# Eulerian-Video-Magnification


<p>Eulerian video magnification reveals temporal variations in videos that are difficult or impossible to see with the naked eye and display them in an indicative manner.
</p>
<p>We are going to create a system which takes in an input video and outputs a video that is motion magnified. The system first decomposes the input video sequence into different
spatial frequency bands, and applies the same temporal filter to all bands. The filtered spatial bands are then amplified by a given factor $\alpha$,
added back to the original signal, and collapsed to generate the output video.</p> 

## Overview of the Eulerian video magnification framework


![](https://github.com/joeljose/assets/blob/master/EVM/EVM_flow.png?raw=True)

## There are 5 steps in the algorithm pipeline:
1) Loading the video</br> 
2) Spatial decomposition into laplacian pyramids</br>
3) Temporal filtering to extract motion information, and adding that back to the original signal</br>
4) Reconstruction </br>
5) Saving to output video</br>

## Downloading the input video to our notebook
To do a demo of our motion magnification algorithm we use a video which was used in the original paper. </br>
You can alternatively use you own video as input too. You will have to upload the video to the colab notebook, and rename the 'filename' variable as the name of your video.</br>

To amplify motion, EVM does not perform feature
tracking or optical flow computation, but merely magnifies temporal color changes using spatio-temporal processing. This Eulerian based method, which temporally processes pixels in a fixed spatial
region, successfully reveals informative signals and amplifies small motions in real-world videos.

One drawback of this method is that we can see that we do artifacts in our videos as we increase amplification factor.

The Algorithm we use is derived from MIT CSAIL's paper, ["Eulerian Video Magnification for Revealing Subtle Changes in the World"](http://people.csail.mit.edu/mrub/papers/vidmag.pdf). I have implemented their paper using Python.

You can also [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joeljose/Eulerian-Video-Magnification/blob/main/Eulerian_Video_Magnification.ipynb)



## Follow Me
<a href="https://twitter.com/joelk1jose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/tw.png" width="30"></a>
<a href="https://github.com/joeljose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/gthb.png" width="30"></a>
<a href="https://www.linkedin.com/in/joel-jose-527b80102/" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/lnkdn.png" width="30"></a>

<h3 align="center">Show your support by starring the repository 🙂</h3>
