---
title: Multi-dimensional image visualization
excerpt: Color-coded representations of time or depth dimensions in a single image with a script written in MATLAB.
header:
  image: /assets/multidem_color/CMZ_DGcolorCodedRep_A.png
  teaser: /assets/multidem_color/CMZ_DGcolorCodedRep_A.png
gallery:
  - url: /assets/multidem_color/080229b9_compositefilt.png
    image_path: /assets/multidem_color/080229b9_compositefilt.png
    alt: "Astrocytic calcium wave along a blood vessel. From Lacar et al., 2011, EJN." 
  - url: /assets/multidem_color/F8.large_edit.jpg
    image_path: /assets/multidem_color/F8.large_edit.jpg
    alt: "Color assignment based on reaching threshold signal. From Lacar et al., 2012, J. Neurosci."

---

This was the script that was used to produce the [cover image](https://benslack19.github.io/_pages/about-the-cover/). In some ways, it was my first "data science" project since it involved programming (it's written in MATLAB) and statistics. I wrote it while I was in graduate school, studying the effect of astrocytic calcium waves on blood vessel diameter. The project involved taking a lot of time-lapse images to study both the changes in activity and changes in the vasculature. (If you're not a neuroscientist, here's a little bit more explanation. Whereas most people associate neurons with the brain, glia are the "other" main cell type, which actually outnumber neurons. They typically play supporting functions, including regulation of blood vessel diameter. My thesis project looked at how activity in a subclass of glia can influence blood flow in a part of the brain called the subventricular zone.)

Anyway, I looked at a lot of time-lapse images trying to see patterns of activity. I would look at them in various ways--clicking through them one-by-one, playing them at different speeds in a movie, watching the images backwards, etc. One day I attended a talk where someone encoded three different time points red, blue, and green. I though, "What if you apply the same idea but use more time points and more colors?" Of course, you had to make sure that the signal did not repeat itself in the same place, but most of the activity I was looking at was in the form of waves. One example is shown in the first figure here.

{% include gallery caption="Examples of activity waves color-coded by timepoint. The first figure shows activity along a blood vessel. The second figure shows how the timepoints of activity become assigned according to the color map." %}

When paired with a scale bar, it was easy to tell things like how fast the wave moved and how far it traveled. I thought it was a cool way to show dynamic things in a static image and so I therefore applied it to other examples. It wasn't long before I realized it can be applied to a different kind of image series, one where the represented dimension was depth instead of time. Those are the examples on the [home page](https://benslack19.github.io/) and the banner of <a href="#top">this page</a>.

The strategy to encode color based on image series position was the following. For each pixel, I would determine the average signal intensity across the image series. Then I would define a threshold for which to assign that pixel a color value (as shown in the second figure). This threshold would be above the average, either a percent change or relative to the standard deviation. The idea is that the first time the pixel reaches signal at its threshold, it will be assigned the color that maps to that point in the image series. The pixels would then get merged to produce the resulting images.

One factor influencing the number of images is how much the signal "moves" in the image series. The more the signal moves, the more resolution the image series dimension can be represented by color.

The MATLAB code for this script can be found on my Github page [here](https://github.com/benslack19/multidim-color-representation).
