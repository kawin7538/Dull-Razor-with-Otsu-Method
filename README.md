# Dull-Razor-with-Otsu-Method
execute Dull razor algorithm with otsu method and play animation
## From Original Image

![input_image](https://user-images.githubusercontent.com/20518062/120342079-a38a0280-c321-11eb-81a6-5087b7fa28f3.jpeg)

## convert into grayscale

![input_gray](https://user-images.githubusercontent.com/20518062/120342138-b43a7880-c321-11eb-9657-7c38506a1288.jpg)

## compute otsu threshold 
- within class variance graph
![withinClassVariance](https://user-images.githubusercontent.com/20518062/120342276-d7fdbe80-c321-11eb-99f6-946dda89464a.png)

## binary mask from optimal threshold
![output_binarymask](https://user-images.githubusercontent.com/20518062/120342449-ff548b80-c321-11eb-9b27-3c047815947b.jpg)

## Next, find contour from the binary mask above
- image with all contours

![image_w_all_contours](https://user-images.githubusercontent.com/20518062/120342631-2b700c80-c322-11eb-8226-311a67c502c4.jpg)

## Last, find biggest contour (as lesion image)
![image_w_contour](https://user-images.githubusercontent.com/20518062/120342716-3fb40980-c322-11eb-8687-cc4051127e1b.jpg)

crop image using BITWISE AND operator between image contour floodfill and zeros ndarray
![cropped_result](https://user-images.githubusercontent.com/20518062/120342872-640fe600-c322-11eb-842c-dc23c8dea08f.jpg)
