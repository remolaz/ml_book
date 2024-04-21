# Detection Algorithms

## Object Localization

Object detection is one of the areas of computer vision that's just exploding and is working so much better than just a couple of years ago.

### Classification, localization and detection problems

![image](img/ch_21/img_1.jpg)

In order to build up to object detection, you first learn about object localization.
You're already familiar with the image classification task where an algorithm looks at a picture and might be responsible for saying if it is a car. So that is classification.

The problem we are facing here is classification with localization. Which means that the algorithm has to label the picture as car and then is also responsible for putting a bounding box, or drawing a red rectangle around the position of the car in the image.

So that's called the classification with localization problem. Where the term localization refers to figuring out where in the picture is the car you've detected. The classification and the classification with localization problems usually have one object: one big object in the middle of the image that you're trying to recognize or recognize and localize.

In contrast, in the detection problem there can be multiple objects of different categories within a single image.

So the ideas about image classification will be useful for classification with localization. And that the ideas for localization will then turn out to be useful for detection.

### The classification with localization problem

![image](img/ch_21/img_2.jpg)

With the image classification problem, you might input a picture into a ConvNet with multiple layers and this results in a vector features that is fed to a softmax unit that outputs the predicted class. So if you are building a self driving car, your object categories might be the following: a pedestrian, or a car, or a motorcycle, or a background. So if there's no pedestrian, no car, no motorcycle, then you might have an output background. So these are your classes, they have a softmax with four possible outputs. So this is the standard classification pipeline.

How about if you want to localize the car in the image as well. To do that, you can change your neural network to have a few more output units that output a bounding box. So, in particular, you can have the neural network output four more numbers: bx, by, bh, and bw; and these four numbers parameterized the bounding box of the detected object. So in these videos, I am going to use the notational convention that the upper left of the image, has the coordinate (0,0), and at the lower right is (1,1).

So, specifying the bounding box, the red rectangle requires specifying the midpoint. So that’s the point bx, by as well as the height, that would be bh, as well as the width, bw of this bounding box. So now your training set contains not just the object class label, which a neural network is trying to predict up here, but it also contains four additional numbers: giving the bounding box then you can use supervised learning to make your algorithm outputs not just a class label but also the four parameters to tell you where is the bounding box of the object you detected.

So in this example the ideal bx might be about 0.5 because this is about halfway to the right to the image. bhy might be about 0.7 since it's about maybe 70% to the way down to the image. bh might be about 0.3 because the height of this red square is about 30% of the overall height of the image. And bw might be about 0.4 let's say because the width of the red box is about 0.4 of the overall width of the entire image.

![image](img/ch_21/img_3.jpg)

So let's formalize this a bit more in terms of how we define the target label y for this as a supervised learning task.

So we have four classes, and the neural network outputs those four numbers as probabilities of the class labels.

So, let's define the target label y as follows.

$y=[P_c, bx, by, bh, bw, c_1, c_2, c_3]$

Is going to be a vector where the first component Pc tells if there is an object: it represents the probability that there's an object. So, if the object is, class 1, 2 or 3, Pc will be equal to 1. And if it's the background class, so if it's none of the objects you're trying to detect, then Pc will be 0.

Next if there is an object, then you wanted to output bx, by, bh and bw, the bounding box for the object you detected. And finally if there is an object, so if Pc is equal to 1, you wanted to also output c1, c2 and c3 which tells us if the detected object belongs to the class 1, class 2 or class 3: so pedestrian, car or a motorcycle.

**Remember in the problem we're addressing we assume that the input image has only one object.**

So let's go through a couple of examples. If in the trainining image there is a car, the output vector will be

$y = [1, bx, by, bh, bw, 0, 1, 0]$

Pc will be equal to 1 because there is an object, then bx, by, by, bh and bw will specify the bounding box. Then c1 will be 0 because it's not a pedestrian, c2 will be 1 because it is car, c3 will be 0 since it is not a motorcycle.

What if there's no object in the image?

$y = [0, ?, ?, ?, ?, ?, ?, ?]$

In this case, Pc would be equal to 0, and the rest of the elements, will be don't cares, so I'm going to write question marks in all of them. So this is a don't care, because if there is no object in this image, then you don't care what bounding box the neural network outputs as well as which of the three objects, c1, c2, c3 it thinks it is.

So given a set of label training examples, this is how you will construct x, the input image as well as y, the cost label both for images where there is an object and for images where there is no object. And the set of this will then define your training set.

## Loss function

Finally, next let's describe the loss function you use to train the neural network. So the ground true label was y and the neural network outputs some ŷ. What should be the loss be? Well if you're using squared error then the loss can be:

$L(ŷ,y) = (ŷ_1 - y_1)^2 + (ŷ_2 - y_2)^2 + ... + (ŷ_8 - y_8)^2 \text{ if } y_1 = 1$

Notice that y here has eight components as described before. So the loss function goes from the sum of the squares of the difference of the elements. So y1 = Pc, and if pc = 1, it means that there is an object in the image and then the loss can be the sum of squares of all the different elements.

The other case is if y1=0, so that's if this pc = 0.

$L(ŷ,y) = (ŷ_1 - y_1)^2 \text{ if } y_1 = 0$

In that case the loss can be just (ŷ1-y1) squared, because we don't care about all of the rest of the components, so all you care about is how accurately is the neural network ourputting pc in that case.

So just a recap, if y1 = 1, then you can use squared error to penalize square deviation from the predicted, and the actual output of all eight components. Whereas if y1 = 0, then from the second to the eighth components I don't care: all you care about is how accurately is your neural network estimating y1, which is equal to pc.

### Note

I've used the squared error just to simplify the description here. In practice you could probably use a log like feature loss for the c1, c2, c3 of the softmax output. For the bounding box coordinates you can use squared error or something like squared error. For Pc you could use something like the logistics regression loss, although even if you use squared error it'll probably work okay.

### Conclusion

So that's how you get a neural network to not just classify an object but also to localize it. The idea of having a neural network output a bunch of real numbers to tell you where things are in a picture turns out to be a very powerful idea.

## Landmark Detection

![image](img/ch_21/img_4.jpg)

Previously, you saw how you can get a neural network to output four numbers of bx, by, bh, and bw to specify the bounding box of an object you want a neural network to localize.

In more general cases, you can have a neural network just output X and Y coordinates of important points, sometimes called image landmarks, that you want the neural networks to recognize. Let me show you a few examples.

Let's say you're building a face recognition application and for some reason, you want the algorithm to tell you where is the corner of someone's eye: so that point has an X and Y coordinate. So you can have a neural network where its final layer outputs two more numbers: lx and ly that represent the coordinates of that corner of the person's eye.

Now, what if you want the network to tell you all four corners of the eyes, then you could modify the neural network to output l1x, l1y for the first point and l2x, l2y for the second point and so on, so that the neural network can output the estimated position of all those four points of the person's face.

But what if you need some key points along the mouth? You can extract the mouth shape and tell if the person is smiling or frowning. You can also define 64 points or 64 landmarks on the face, that help you define the edge of the face, defines the jaw line, etc. Having these landmarks points means to have a corresponding labeled training set. And here I'm using l to stand for a landmark. So this example you would have 129 output units, one for is a face or not? And then if you have 64 landmarks (couples of x, y coordinates), that's sixty-four times two.

So, this is a basic building block for recognizing emotions from faces and if you played with the Snapchat and the other entertainment, also AR augmented reality filters like the Snapchat photos can draw a crown on the face and have other special effects like putting a crown or a hat on the person.

Of course, in order to treat a network like this, you will need a label training set. We have a set of images as well as labels Y where people will have had to go through and laboriously annotate all of these landmarks.

One last example, if you are interested in people pose detection, you could also define a few key positions like the midpoint of the chest, the left shoulder, left elbow, the wrist, and so on, and just have a neural network to annotate key positions in the person's pose to output the pose of the person. And of course, to do that you also need to specify these key landmarks like maybe l1x and l1y is the midpoint of the chest down to maybe l32x, l32y, if you use 32 coordinates to specify the pose of the person.

So, this idea might seem quite simple of just adding a bunch of output units to output the X,Y coordinates of different landmarks you want to recognize. To be clear, the identity of landmarks must be consistent across different images like maybe landmark one is always this corner of the eye, landmark two is always this corner of the eye, landmark three, landmark four, and so on. So, the labels have to be consistent across different images.

But if you can hire labelers or label yourself a big enough data set to do this, then a neural network can output all of these landmarks which is going to used to carry out other interesting effect such as with the pose of the person, maybe try to recognize someone's emotion from a picture, and so on.

## Object Detection

### Object detection using the Sliding Windows Detection Algorithm

You've learned about Object Localization as well as Landmark Detection. Now, let's use a ConvNet to perform object detection using something called the Sliding Windows Detection Algorithm.

![image](img/ch_21/img_5.jpg)

Let's say you want to build a car detection algorithm. You can first create a label training set with closely cropped examples of cars as in figure. For our purposes in this training set, there are only images with cars that are closely cropped. Meaning that x (the input image) is pretty much only the car. So, you can take a picture and crop out and just cut out anything else that's not part of a car: you end up with the car centered in pretty much the entire image. So, in the training set there are iamges where a car is present and images where the car is not there. Given this label training set, you can then train a ConvNet that inputs an image and then output y as zero or one, if there is a car or not. Once you've trained up this ConvNet you can then use it in Sliding Windows Detection.  

![image](img/ch_21/img_6.jpg)

The Sliding Windows Detection algorithm works in this way: you select a window size to crop your input image. Then you take your input image and you divide it into windows. After you would input the windowed image into your trained ConvNet. THe model will then predict for that small rectangular region, if a car is present or not.

In the Sliding Windows Detection Algorithm, what you do is you then pass as input a second image now bounded by the selected window but shifted a little bit over and feed that to the ConvNet, and run the ConvNet again. And then you do that with a third image and so on. And you keep going until you've slid the window across every position in the image.

The idea is to pass lots of little cropped images into the ConvNet and have it classified zero or one for each position with some stride.

#### Then use larger windows

You then repeat it, but now use a larger window. So, now you take a slightly larger region and run that region. So, resize this region into whatever input size the ConvNet is expecting, and feed that to the ConvNet and have it output zero or one. And then slide the window over again using some stride and so on. And you run that throughout your entire image until you get to the end.

And then you might do the third time using even larger windows and so on.

So this algorithm is called Sliding Windows Detection because you take these windows, these square boxes, and slide them across the entire image and classify every square region with some stride as containing a car or not.

Now there's a huge disadvantage of Sliding Windows Detection, which is the computational cost. Because you're cropping out so many different square regions in the image and running each of them independently through a ConvNet. And if you use a very coarse stride, a very big stride, a very big step size, then that will reduce the number of windows you need to pass through the ConvNet, but that courser granularity may hurt performance. Whereas if you use a very fine granularity or a very small stride, then the huge number of all these little regions you're passing through the ConvNet means that means there is a very high computational cost.

So, before the rise of Neural Networks people used to use much simpler classifiers like a simple linear classifier over hand engineer features in order to perform object detection. And in that era because each classifier was relatively cheap to compute (it was just a linear function), the Sliding Windows Detection ran okay. It was not a bad method, but with ConvNet now running a single classification task is much more expensive and sliding windows this way is infeasibily slow.

And unless you use a very fine granularity or a very small stride, you end up not able to localize the objects that accurately within the image as well. Fortunately however, this problem of computational cost has a pretty good solution. In particular, the Sliding Windows Object Detector can be implemented convolutionally or much more efficiently.

## Convolutional Implementation of Sliding Windows

In the last section, we learned about the sliding windows object detection algorithm using a ConvNet but we saw that it was too slow. In this section, we will learn how to implement that algorithm convolutionally.

### Turn fully connected layers into convolutional layers

![image](img/ch_21/img_7.jpg)

Let's first see how you can turn fully connected layers in neural network, into convolutional layers.

Let's say that your object detection algorithm inputs 14 by 14 by 3 images. It then uses 16 filters with 5 by 5 by 3 size: in this way the input image is mapped to 10 by 10 by 16.

Then it does a 2 by 2 max pooling to reduce the input size to 5 by 5 by 16.

Then there is a fully connected layer that connects 400 units. Another fully connected layer with 400 units, and then finally the model outputs the prediction Y using a softmax unit. The output layer has 4 units, corresponding to the class probabilities of the four classes that softmax units is classified amongst. And the four classes could be pedestrian, car, motorcycle, and background.

Now, let's see how these layers can be turned into convolutional layers.

So, the convnet will draw same as before for the first few layers.

Next we use 400 filters of size 5 by 5 by 16 that will map the output of the max pooling layer to 1 by 1 by 400.

Remember, a 5 by 5 filter is implemented as 5 by 5 by 16 because our convention is that the filter looks across all 16 channels. So the 3rd input dimension 16 and the filter 3rd dimension 16 must match and so the outputs will be 1 by 1. And if you have 400 of these 5 by 5 by 16 filters, then the output dimension is going to be 1 by 1 by 400.

So rather than viewing these 400 as just a set of nodes, we're going to view this as a 1 by 1 by 400 volume. Mathematically, this is the same as a fully connected layer because each of these 400 nodes has a filter of dimension 5 by 5 by 16. So each of those 400 values is some arbitrary linear function of these 5 by 5 by 16 activations from the previous layer.

Next, to implement the next convolutional layer, we're going to implement a 1 by 1 convolution. If you have 400 1 by 1 filters then, with 400 filters the next layer will again be 1 by 1 by 400. So that gives you the next fully connected layer.

And then finally, we're going to have another 1 by 1 by 400 filter, followed by a softmax activation. So as to give a 1 by 1 by 4 volume to give predictions.

So this shows how you can take these fully connected layers and implement them using convolutional layers.

### Convolutional implementation of sliding windows object detection

![image](img/ch_21/img_8.jpg)

After this conversion, let's see how you can have a convolutional implementation of the sliding windows object detection. The presentation on the image is based on the OverFeat paper, by Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Robert Fergus and Yann Le Cunn.

Let's say that your sliding windows ConvNet inputs 14 by 14 by 3 images and again, and that eventually outputs a 1 by 1 by 4 volume, which is the output of your softmax.

To simplify the drawing in the image we cut the third dimension, in fact the input image should be a volume of 14 by 14 by 3 is technically a volume and so the 16 filters with size 5 by 5 by 3, and so the output 10 by 10 by 16, and the 1 by 1 by 400 volume, and so on... To simplify the drawing for this slide, we just draw the front face of these volumes.

Let's say that your ConvNet inputs 14 by 14 by 3 images and your tested image is 16 by 16 by 3. So we added that yellow stripe to the border of the test image. In the original sliding windows algorithm, you might want to input the blue region into a ConvNet and run that once to generate a classification 0 or 1 and then slightly down a bit, let's use a stride of two pixels and then you might slide that to the right by two pixels to input this green rectangle into the ConvNet and we run the whole ConvNet and get another label, 0 or 1. Then you might input this orange region into the ConvNet and run it one more time to get another label. And then do it the fourth and final time with this lower right purple square. You run this ConvNet four times in order to get four labels.

But it turns out a lot of this computation done by these four convnets is highly duplicative. So what the convolutional implementation of sliding windows does is it allows these four passes in the ConvNet to share a lot of computation. Specifically, here's what you can do.

You can take the ConvNet and just run it using the same parameters but taking in input the whole test image that has shape 16 by 16 by 3. So you use 16 filters of size 5 by 5 by 3 and you run the first convolution. Now, we have a 12 by 12 by 16 output volume. Then do the max pool, same as before. Now we have a 6 by 6 by 16 volume, that runs through 400 filters with size 5 by 5 by 16 to get an output volume of 2 by 2 by 400. We run it through 400 filters of size 1 by 1 and gives you another 2 by 2 by 400. Do that one more time and now we are left with a 2 by 2 by 4 output volume.

Considering the output volume 2 by 2 by 4 we obtained, it turns out that the upper left corner, the blue 1 by 1 by 4 subset gives you the result of running the ConvNet on the upper left hand corner 14 by 14 by 3 of the input image that has size 16 by 16 by 3. Instead, this upper right 1 by 1 by 4 volume gives you the upper right result. The lower left gives you the results of implementing the ConvNet on the lower left 14 by 14 region. And the lower right 1 by 1 by 4 volume gives you the same result as running the ConvNet on the lower right 14 by 14 medium.

And if you step through all the steps of the calculation, let's look at the green region in the example, if you had cropped out just this region and passed it through the ConvNet, then the first layer's activations would have been exactly this region. The next layer's activation after max pooling would have been exactly this region and then the next layer, the next layer would have been as follows.

So what this convolution implementation does is, instead of forcing you to run four propagation on four subsets (14 by 14 by 3 patches) of the input image independently, it combines all four into one form of computation and shares a lot of the computation in the regions of image that are common.

![image](img/ch_21/img_9.jpg)

At 10:04 onward, the size of the second layer should be 24 x 24 instead of 16 x 16.

Now let's just go through a bigger example. Let's say you now want to run sliding windows on a 28 by 28 by 3 image. It turns out that if you run it in the same way then you end up with an 8 by 8 by 4 output. And that corresponds to running a sliding windows on the first 14 by 14 by 3 subset region thus, giving you the output corresponding the upper left hand corner. Then using you can use a slider to shift one window over, one window over, one window over and so on and the eight positions. So that gives you this first row and then as you go down the image as well, that gives you all of these 8 by 8 by 4 outputs. Because of the max pooling of 2, that corresponds to running your neural network with a stride of two on the original image.

So just to recap, to implement sliding windows, previously, what you do is you crop out a region. Let's say this is 14 by 14 and run that through your ConvNet and do that for the next region over, then do that for the next 14 by 14 region, then the next one, then the next one, and so on, until hopefully that one recognizes the car.

But now, instead of doing it sequentially, with this convolutional implementation that you saw in the previous image, you can implement the entire image, all maybe 28 by 28 by 3 and convolutionally make all the predictions at the same time by one forward pass through this big ConvNet and hopefully have it recognize the position of the car.

So that's how you implement sliding windows convolutionally and it makes the whole thing much more efficient. Now, this algorithm still has one weakness, which is the position of the bounding boxes is not going to be too accurate.

## Bounding Box Predictions (starting YOLO)

In the last video, you learned how to use a convolutional implementation of sliding windows. That's more computationally efficient, but it still has a problem of not quite outputting the most accurate bounding boxes.

![image](img/ch_21/img_10.jpg)

Now let's see how you can get your bounding box predictions to be more accurate. With sliding windows, you take this three sets of locations and run the classifier through it. And in this case, none of the boxes really match up perfectly with the position of the car. So, maybe that box is the best match. And also, it looks like the perfect bounding box isn't even square, it's actually a rectangle with horizontal aspect ratio.

So, is there a way to get this algorithm to outputs more accurate bounding boxes? A good way to get this output more accurate bounding boxes is with the YOLO algorithm. YOLO stands for, You Only Look Once. And is an algorithm due to Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi. Here's what you do.

Let's say you have an input image at 100 by 100, you're going to place down a grid on this image. And for the purposes of illustration, I'm going to use a 3 by 3 grid. Although in an actual implementation, you use a finer one, a 19 by 19 grid. And the basic idea is you're going to take the image classification and localization algorithm that you saw earlier and apply that to each of the nine grid cells of this image.

So to be more concrete, here's how you define the labels you use for training. For each of the nine grid cells, you specify a label Y, where the label Y is this eight dimensional vector, same as you saw previously.

$y=[P_c, bx, by, bh, bw, c_1, c_2, c_3]$

Your first output $P_c$ can be  or 1 depending on whether or not there's an object in that grid cell and then BX, BY, BH, BW to specify the bounding box if there is an object associated with that grid cell. And then say, C1, C2, C3, if you try and recognize three classes not counting the background class. So you try to recognize pedestrian's class, motorcycles and the background class: then C1 C2 C3 can be the pedestrian, car and motorcycle classes.

So in this image, we have nine grid cells, so you have a vector like this for each of the grid cells.

To give a bit more detail, this image has two objects. And what the YOLO algorithm does is it takes the midpoint of reach of the two objects and then assigns the object to the grid cell containing the midpoint. So the left car is assigned to this grid cell, and the car on the right, which has that midpoint, is assigned to that other grid cell. And so even though the central grid cell has some parts of both cars, we'll pretend the central grid cell has no interesting object.

So let's start with the upper left grid cell, this one up here. For that one, there is no object. So, the label vector Y for the upper left grid cell would be zero, and then don't cares for the rest of these $y = [0, ?, ?, ?, ?, ?, ?, ?]$ The output label Y would be the same for all the grid cells with nothing, with no interesting object in them. The same for the central grid cell that contains no midpoint.

Now, how about this grid cell where there is in fact a car in it?

Whereas for this cell, this cell that I have circled in green on the left, the target label Y would be as follows.

$y=[1, bx, by, bh, bw, 0, 1, 0]$

There is an object, and then you write BX, BY, BH, BW, to specify the position of this bounding box. And then you have: class one was a pedestrian, then that was zero; class two is a car, that's one; class three was a motorcycle, that's zero. And then similarly, for the grid cell on the right because that does have an object in it, it will also have some vector like this as the target label corresponding to the grid cell on the right.

So, for each of these nine grid cells, you end up with a eight dimensional output vector Y. And because you have 3 by 3 grid cells, you have nine grid cells, the total volume of the output is going to be 3 by 3 by 8.

So now, to train your neural network, the input is 100 by 100 by 3, that's the input image. And then you have a usual ConvNet with conv layers, max pool layers, and so on, that eventually maps to a 3 by 3 by 8 output volume. So what you do is you have an input X which is the input image, and you have these target labels Y which are 3 by 3 by 8, and you use back propagation to train the neural network to map from any input X to this type of output volume Y.

The advantage of this algorithm is that the neural network outputs precise bounding boxes. So at test time, what you do is you feed an input image X and run forward prop until you get this output Y. And then for each of the nine outputs of each of the 3 by 3 positions, you can then just read if there is an object associated with that part of the grid or not. And so long as you don't have more than one object in each grid cell, this algorithm should work okay.

And the problem of having multiple objects within the grid cell is something we'll address later.

We use a relatively small 3 by 3 grid, in practice, you might use a much finer, grid maybe 19 by 19. So you end up with 19 by 19 by 8, and that also makes your grid much finer. It reduces also the chance that there are multiple objects assigned to the same grid cell.

#### Reminder about YOLO

The way you assign an object to grid cell as you look at the midpoint of an object and then you assign that object to whichever one grid cell contains the midpoint of the object. So even if the objects spends multiple grid cells, that object is assigned only to one of the nine grid cells, or one of the 19 by 19 grid cells.

So notice two things:

1. This is a lot like the image classification and localization algorithm with the addition that it outputs the bounding box coordinates explicitly. This allows your network to output bounding boxes of any aspect ratio, as well as, output much more precise coordinates that aren't just dictated by the stripe size of your sliding windows classifier.
2. This is a convolutional implementation and you're not implementing this algorithm nine times on the 3 by 3 grid or 361 times if you're using a 19 by 19 grid. Instead, this is one single convolutional implantation, where you use one ConvNet with a lot of shared computation between all the computations needed for all of your 3 by 3 or all of your 19 by 19 grid cells. So, this is a pretty efficient algorithm. And in fact, one nice thing about the YOLO algorithm, which is constant popularity is because this is a convolutional implementation, it actually runs very fast. So this works even for real time object detection.

#### How do you encode these bounding boxes [bx, by, bh, bw]?

![image](img/ch_21/img_11.jpg)

Let's take the example of the car on the right. So, in this grid cell there is an object and so in the target label y P_c is equal to one. And then bx, by, bh, bw, and then 0 1 0. So, how do you specify the bounding box?

In the YOLO algorithm, I take the convention that the upper left point of the grid box we are considering is (0,0) and the lower right point is (1,1). So bx and by coordinates are relative to the specific grid box we are considering in that moment. So to specify the position of the car midpoint (that orange dot) in the square grid on the right, bx might be 0.4. And then y, looks I guess maybe 0.3. And then the height of the bounding box is specified as a fraction of the overall width of this box. So, the width of this red bounding box is maybe 90% of that blue line (width of the grid box), so bw is 0.9. And so bh is 0.5: one half of the overall height of the grid cell.

So, this bx, by, bh, bw are specified relative to the grid cell: so bx and by has to be between 0 and 1. Because pretty much by definition that orange dot is within the bounds of that grid cell is assigned to. If it wasn't between 0 and 1 it was outside the square, then we'll have been assigned to a different grid cell.

But bw and bh could be greater than one. In particular if you have a car where the bounding box was that, then the height and width of the bounding box, this could be greater than one.

So, there are multiple ways of specifying the bounding boxes, but this would be one convention that's quite reasonable. Although, if you read the YOLO research papers, there are other parameterizations that work even a little bit better, but I hope this gives one reasonable condition that should work okay.

There are some more complicated parameterizations involving sigmoid functions to make sure bx and by is between 0 and 1. And they also used other parameterizations to make sure that bw and bh are non-negative, these have to be greater or equal to zero. There are some other more advanced parameterizations that work things a little bit better, but the one you saw here should work okay.

The YOLO paper is one of the harder papers to read.

## Intersection over Union

![image](img/ch_21/img_12.jpg)

So how do you tell if your object detection algorithm is working well? THere is a function called, "Intersection Over Union".

In the object detection task, you expected to localize the object as well. So if the red box is the ground-truth bounding box, and if your algorithm outputs this bounding box in purple, how can we say if this is a good outcome or a bad one?
So what the intersection over union function does, or IoU does, is it computes the intersection over union of these two bounding boxes.

The union of these two bounding boxes is the area that is contained in either bounding boxes; whereas the intersection is the area in common to both red and purple boxes. So what the intersection over union does: it computes the size of the intersection, that orange shaded area, and divided by the size of the union, which is that green shaded area.

By convention, this divions will judge if your answer is correct if the IoU is greater than 0.5. If the predicted and the ground-truth bounding boxes overlapped perfectly, the IoU would be one, because the intersection would equal to the union. But in general, so long as the IoU is greater than or equal to 0.5, then the answer will look pretty decent. The higher the IoUs, the more accurate the bounding the box.

This is one way to map localization. And again 0.5 is just a human chosen convention. There's no particularly deep theoretical reason for it. You can also choose some other threshold like 0.6 if you want to be more stringent. I rarely see people drop the threshold below 0.5.

Now, what motivates the definition of IoU, as a way to evaluate whether or not your object localization algorithm is accurate or not. More generally, IoU is a measure of the overlap between two bounding boxes. Where if you have two boxes, you can compute the intersection, compute the union, and take the ratio of the two areas. And so this is also a way of measuring how similar two boxes are to each other.

## Non-max suppression

![image](img/ch_21/img_13a.jpg)

One of the problems of Object Detection, is that your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times. Non-max suppression is a way for you to make sure that your algorithm detects each object only once. Let's go through an example.

Let's say you want to detect pedestrians, cars, and motorcycles in this image. You might place a 19 by 19 grid. Now, while technically this car has just one midpoint, so it should be assigned just one grid cell. And the car on the left also has just one midpoint, so technically only one of those grid cells should predict that there is a car. In practice, you're running an object classification and localization algorithm for every one of these split cells. So it's quite possible that this split cell might think that the center of a car is in it, and so might the split cell next to it, and so might another one, and for the car on the left as well.

Let's step through an example of how non-max suppression will work. So, because you're running the image classification and localization algorithm on every grid cell, on 361 grid cells (19 by 19), it's possible that many of them will raise their hand and say, "My P_c, my chance of thinking I have an object in it is large." So, when you run your algorithm, you might end up with multiple detections of each object. So, what non-max suppression does, is it cleans up these detections. So they end up with just one detection per car, rather than multiple detections per car.

Concretely, what it does:

1. it first looks at the probabilities associated with each of these detections: P_c. It actually considers P_c times C1, or C2, or C3. But for now, let's just say it considers P_c as the probability of a detection. And it first takes the largest P_c, which in this case is 0.9 and says, "That's my most confident detection, so let's highlight that and just say I found the car there."
2. Having done that the non-max suppression part then looks at all of the remaining rectangles and all the ones with a high overlap (with a high IoU) with the one with P_c=0.9, will get suppressed. So those two rectangles with P_c 0.6 and P_c 0.7 that overlap a lot with the rectangle P_c=0.9 will be suppressed.
3. Next, you then go through the remaining rectangles and find the one with the highest probability, the highest Pc, which in this case is this one with 0.8. And then, the non-max suppression part is to get rid of any other ones with a high IOU with it.
4. So now, every rectangle has been either highlighted or deleted, and these are your two final predictions.

So, this is non-max suppression. And non-max means that you're going to output your maximal probabilities classifications but suppress the close-by ones that are non-maximal. Hence the name, non-max suppression.

![image](img/ch_21/img_13b.jpg)

Let's go through the details of the algorithm. First, on this 19 by 19 grid, you're going to get a 19 by 19 by eight output volume. For this example, I'm going to simplify it to say that you only doing car detection. So, let me get rid of the C1, C2, C3, and pretend that for each of the 361 positions, you get an output prediction of the following: the chance there's an object, and then the bounding box. And because you have only one object, there's no C1, C2, C3 prediction.

1. Now, to intimate non-max suppression, the first thing you can do is discard all the boxes, discard all the predictions of the bounding boxes with P_c less than or equal to some threshold, let's say 0.6. So for each of the 361 positions, you output a bounding box together with a probability of that bounding box being a good one. So we're just going to discard all the bounding boxes that were assigned a low probability.

2. Next, while there are any remaining bounding boxes that you've not yet discarded or processed, you're going to repeatedly pick the box with the highest probability, with the highest P_c, and then output that as a prediction. So you commit to outputting that as a prediction for that there is a car there.

3. Next, you then discard any remaining box with a high overlap, with a high IOU, with the box that you just output in the previous step. And so, you keep doing this while there's still any remaining boxes that you've not yet processed, until you've taken each of the boxes and either output it as a prediction, or discarded it as having too high IOU, with one of the boxes that you have just output as your predicted position for one of the detected objects.

I've described the algorithm using just a single object on this slide. If you actually tried to detect three objects say pedestrians, cars, and motorcycles, then the output vector will have three additional components. And it turns out, the right thing to do is to independently carry out non-max suppression three times, one on each of the outputs classes.

## Anchor boxes

One of the problems with object detection as you have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects? You can use the idea of anchor boxes. Let's start with an example.

![image](img/ch_21/img_14.jpg)

Let's say you have an image like this. And for this example, I am going to continue to use a 3 by 3 grid. Notice that the midpoint of the pedestrian and the midpoint of the car are in almost the same place and both of them fall into the same grid cell. So, for that grid cell, if Y outputs this vector

$y = [P_c, bx, by, bh, bw, c1, c2, c3]$

where you are detecting three classes, pedestrians, cars and motorcycles, it won't be able to output two detections. So I have to pick one of the two detections to output.

With the idea of anchor boxes, what you are going to do, is pre-define two different shapes called, anchor boxes or anchor box shapes. And what you are going to do is now, be able to associate two predictions with the two anchor boxes. And in general, you might use more anchor boxes, maybe five or even more. But for this example, I am just going to use two anchor boxes just to make the description easier. So what you do is you define the label to be:

$y = [P_c, bx, by, bh, bw, c1, c2, c3, P_c, bx, by, bh, bw, c1, c2, c3]$

so you will have PC, PX, PY, PH, PW, C1, C2, C3: eight outputs associated with anchor box 1; and then you repeat other eight outputs associated with anchor box 2. So, because the shape of the pedestrian is more similar to the shape of anchor box 1, you can use these eight numbers to encode that PC as one, yes there is a pedestrian. And then because the box around the car is more similar to the shape of anchor box 2 than anchor box 1, you can then use this to encode that the second object here is the car, and have the bounding box and all the parameters associated with the detected car.

![image](img/ch_21/img_15.jpg)

So to summarize, previously, before using anchor boxes, you did the following: each object in the training set and the training set image, was assigned to the grid cell that corresponds to that object's midpoint. And so the output Y was 3 by 3 by 8 because you have a 3 by 3 grid. And for each grid position, we had that output vector which is PC, then the bounding box, and C1, C2, C3.

With the anchor box, now, each object is assigned to a grid cell and anchor box with the highest IoU with the object's shape. So let's say you have an object with a certain shape, and two anchor boxes: box 1 that has a rectangular vertical shape and box 2 that has a rectangular horizontal shape. Then what you do is take your two anchor boxes and then you see which of the two anchor boxes has a higher IoU with the object shape. And that object then gets assigned not just to a grid cell but to a pair: to grid cell comma anchor box pair. And that's how that object gets encoded in the target label. So now, the output Y is going to be 3 by 3 by 16. Because as you saw on the previous slide, Y is now 16 dimensional. Or if you want, you can also view this as 3 by 3 by 2 by 8, because there are now two anchor boxes and Y is eight dimensional. And dimension of Y being eight was because we have three objects classes; if you have more objects than the dimension of Y would be even higher. So let's go through a complete example.

![image](img/ch_21/img_16.jpg)

For this grid cell, let's specify what is Y. So the pedestrian is more similar to the shape of anchor box 1. So for the pedestrian, we're going to assign it to the top half of this vector (yellow part). So yes, there is an object, there will be some bounding box associated at the pedestrian. And I guess if a pedestrian is class one, then we see c1 as one, and then zero, zero. And then the shape of the car is more similar to anchor box 2. And so the rest of this vector (green part) will be one and then the bounding box associated with the car, and then the car is C2, so there's zero, one, zero. And so that's the label Y for that lower middle grid cell that this arrow was pointing to. Now, what if this grid cell only had a car and had no pedestrian? If it only had a car, then assuming that the shape of the bounding box around the car is still more similar to anchor box 2, then the target label Y will still be the same for the anchor box 2 component, and for the part of the vector corresponding to anchor box 1, you just say there is no object there. So P_c is zero, and then the rest of these will be don't cares.

### What if you have two anchor boxes but three objects in the same grid cell?

This algorithm doesn't have a great way of handling it. I will just influence some default tiebreaker for that case. Or what if you have two objects associated with the same grid cell, but both of them have the same anchor box shape? Again, that's another case that this algorithm doesn't handle well. If you influence some default way of tiebreaking if that happens. Hopefully this won't happen with your data set, and so, it shouldn't affect performance as much.

So, that's it for anchor boxes. And even though I'd motivated anchor boxes as a way to deal with what happens if two objects appear in the same grid cell, in practice, that happens quite rarely, especially if you use a 19 by 19 rather than a 3 by 3 grid. The chance of two objects having the same midpoint in these 361 cells, it does happen that often. Maybe even better results that anchor boxes gives you is it allows your learning algorithm to specialize better. In particular, if your data set has some tall, skinny objects like pedestrians, and some white objects like cars, then this allows your learning algorithm to specialize so that some of the outputs can specialize in detecting wide, fat objects like cars, and some of the output units can specialize in detecting tall, skinny objects like pedestrians.

### How do you choose the anchor boxes?

So finally, how do you choose the anchor boxes? And people used to just choose them by hand or choose maybe five or 10 anchor box shapes that spans a variety of shapes that seems to cover the types of objects you seem to detect.

As a much more advanced version, an even better way to do this in one of the later YOLO research papers, is to use a K-means algorithm, to group together two types of objects shapes you tend to get. And then use that to select a set of anchor boxes that most stereotypically represent the multiple object classes you're trying to detect.

## Q&A on YOLO

How is it possible that some of the bounding boxes can go outside the height and width of the grid cell that they came from?

It's true that in YOLO, some predicted bounding boxes can extend beyond the boundaries of the grid cell they originated from. This happens because:

Grid Cells for Localization, not Size: The YOLO algorithm divides the input image into a grid of cells. Each grid cell is responsible for detecting objects within its boundaries, not necessarily containing the entire object.

Object Center determines Assigned Cell: An object is assigned to the grid cell that contains its center point. This means even if the object itself spans multiple cells, its bounding box information will be associated with the assigned cell.

Bounding Boxes represent Entire Object: The predicted bounding box for an object encompasses the entire object, regardless of which grid cell it falls into.

Therefore, it's natural for bounding boxes to extend beyond the confines of the assigned grid cell, as they represent the object's true extent in the image, not just its presence within a specific cell.

Remo Note: So the ConvNet is taking in input the entire image but just focuses on the cell to localize the center of the objects.

## YOLO Algorithm

![image](img/ch_21/img_17.jpg)

Let's put all the components together to form the YOLO object detection algorithm.

First, let's see how you construct your training set. Suppose you're trying to train an algorithm to detect 3 objects: pedestrians, cars, and motorcycles. If you're using 2 anchor boxes, then the outputs y will be: [3, 3, 2, 8]

- 3 by 3 because you are using 3 by 3 grid cell,
- by 2, this is the number of anchors,
- by 8, that is 5 plus the number of classes [P_c, bx, by, bh, bw, c1, c2, c3]. So five because you have Pc and then the bounding boxes, that's five, and then C1, C2, C3.

And you can either view this as 3 by 3 by 2 by 8, or by 3 by 3 by 16. So to construct the training set, you go through each of these nine grid cells and form the appropriate target vector y.

So in the first grid cell, there's nothing worth detecting. None of the three classes pedestrian, car and motocycle, appear in the upper left grid cell and so, the target y corresponding to that grid cell would be equal to [0, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?]. Where Pc for the first anchor box is zero because there's nothing associated for the first anchor box, and is also zero for the second anchor box and so all of these other values you don't care.

Now, most of the grid cells have nothing in them, but ther eis one with a car inside. So assuming that your training set has two anchor boxes: anchor box one and anchor box 2, being anchor box 2 for the car, just a little bit wider than it is tall.

You make run the ConvNet on the image and the ConvNet outputs the red bounding box in figure.
And so, the red box has just slightly higher IoU with anchor box 2. And so, the car gets associated with this lower portion of the vector Y=[0, ?, ?, ?, ?, ?, ?, ?, 1, bx, by, bh, bw, 0, 1, 0]. So notice then that Pc associate anchor box one is zero. So you have don't cares all these components. Then you have this Pc is equal to one, then you should use these to specify the position of the red bounding box, and then specify that the correct object is class two. Right that it is a car.

So you go through this and for each of your nine grid positions each of your three by three grid positions, you would come up with a vector like this: with a 16 dimensional vector. And so, that's why the final output volume is going to be 3 by 3 by 16. For simplicity on the slide I've used a 3 by 3 the grid. In practice it might be more like a 19 by 19 by 16. Or in fact if you use more anchor boxes, like 5 anchor boxes, it may be 19 by 19 by 5 x 8: so it will be 19 by 19 by 40.

So that's training and you train ConvNet that inputs an image, maybe 100 by 100 by 3, and your ConvNet would then finally output this output volume in our example, 3 by 3 by 16 or 3 by 3 by 2 by 8.

![image](img/ch_21/img_18.jpg)

Next, let's look at how your algorithm can make predictions. Given an image, your neural network will output this by 3 by 3 by 2 by 8 volume, where for each of the nine grid cells you get a vector like Y. So for the grid cell here on the upper left, if there's no object there, hopefully, your neural network will output zero, and zero for P_c, and it will output some other values. Your neural network can't output a question mark, it will put some numbers for the rest. But these numbers will basically be ignored because the neural network is telling you that there's no object there. So it doesn't really matter whether the output there is more or less noise.

In contrast, the output for that box at the bottom left, hopefully would be something like zero for bounding box one, and then just open a bunch of numbers, just noise; then there will be 1 followed by a set of numbers that corresponds to specifying a pretty accurate bounding box for the car. So that's how the neural network will make predictions.

### Run this through non-max suppression

![image](img/ch_21/img_19.jpg)

Finally, you run this through non-max suppression. So just to make it interesting. Let's look at the new test set image. Here's how you would run non-max suppression. If you're using two anchor boxes, then for each of the non-grid cells, you get two predicted bounding boxes. Some of them will have very low probability, very low Pc, but you still get two predicted bounding boxes for each of the nine grid cells. So let's say, those are the bounding boxes you get. And notice that some of the bounding boxes can go outside the height and width of the grid cell that they came from. Next, you then get rid of the low probability predictions. So get rid of the ones that even the neural network says, gee this object probably isn't there. So get rid of those. And then finally if you have three classes you're trying to detect, you're trying to detect pedestrians, cars and motorcycles. What you do is, for each of the three classes, independently run non-max suppression for the objects that were predicted to come from that class. But use non-max suppression for the predictions of the pedestrians class, run non-max suppression for the car class, and non-max suppression for the motorcycle class. But run that basically three times to generate the final predictions. And so, the output of this is hopefully that you will have detected all the cars and all the pedestrians in this image. So that's it for the YOLO object detection algorithm. Which is really one of the most effective object detection algorithms, that also encompasses many of the best ideas across the entire computer vision literature that relate to object detection. And you get a chance to practice implementing many components of this yourself, in this week's problem exercise. So I hope you enjoy this week's problem exercise. There's also an optional video that follows this one which you can either watch or not watch as you please. But either way I also look forward to seeing you next week.

[Interesting link](https://community.deeplearning.ai/t/yolo-library-bounding-box/38717)

## Region proposals

![image](img/ch_21/img_20.jpg)
![image](img/ch_21/img_21.jpg)

If you look at the object detection literature, there's a set of ideas called region proposals that's been very influential in computer vision as well. I tend to use the region proposal set of algorithms a bit less often but nonetheless, it has been an influential body of work and an idea that you might come across in your own work. Let's take a look.

So, if you recall the sliding windows idea, you would take a classifier and run it across all of these different windows and run the detector to see if there's a car, pedestrian, or maybe a motorcycle. Now, you could run the algorithm convolutionally, but one downside is that the algorithm is just classifing a lot of the regions where there's clearly no object. So this rectangle down left here is pretty much blank. It's clearly nothing interesting there to classify.

So what Russ Girshik, Jeff Donahue, Trevor Darrell, and Jitendra Malik proposed in the paper, as cited to the bottom of the image, is an algorithm called R-CNN, which stands for Regions with convolutional networks or regions with CNNs. And what that does is it tries to pick just a few regions that makes sense to run your ConvNet classifier. So rather than running your sliding windows on every single window, you instead select just a few windows, and run your ConvNet classifier on just a few windows.

The way that they perform the region proposals is to run an algorithm called a segmentation algorithm, that results in this output on the right, in order to figure out what could be objects. So, for example, the segmentation algorithm finds a blob over here. And so you might pick that pounding balls and say, "Let's run a classifier on that blob." It looks like this little green thing finds a blob there, as you might also run the classifier on that rectangle to see if there's something interesting there. And in the case of this blue blob, if you run a classifier on that, hope you find the pedestrian, and if you run it on this light cyan blob, maybe you'll find a car, or maybe not.

So this is called a segmentation algorithm, and what you do is you find maybe 2000 blobs and place bounding boxes around about 2000 blobs and value classifier on just those 2000 blobs. This can be a much smaller number of positions on which to run your content classifier, then if you have to run it at every single position throughout the image.

And this is a special case if you are running your ConvNet not just on square-shaped regions but running them on tall skinny regions to try to find pedestrians or running them on your white fat regions try to find cars and running them at multiple scales as well. So that's the R-CNN or the region with CNN.

Now, it turns out the R-CNN algorithm is still quite slow. So there's been a line of work to explore how to speed up this algorithm. So the basic R-CNN algorithm with proposed regions uses some algorithm and then classifies the proposed regions one at a time. And for each of the regions, they will output the label. So is there a car? Is there a pedestrian? Is there a motorcycle there? And then also outputs a bounding box, so you can get an accurate bounding box if indeed there is a object in that region.

So just to be clear, the R-CNN algorithm doesn't just trust the bounding box it was given. It also outputs a bounding box, BX, BY, BH, BW, in order to get a more accurate bounding box and whatever happened to surround the blob that the image segmentation algorithm gave it. So it can get pretty accurate bounding boxes.

Now, one downside of the R-CNN algorithm was that it is actually quite slow. So over the years, there been a few improvements to the R-CNN algorithm. Russ Girshik proposed the fast R-CNN algorithm, and it's basically the R-CNN algorithm but with a convolutional implementation of sliding windows. So the original implementation would actually classify the regions one at a time. So far, R-CNN use a convolutional implementation of sliding windows, and this is roughly similar to the idea you saw previously. And that speeds up R-CNN quite a bit.

It turns out that one of the problems of fast R-CNN algorithm is that the clustering step to propose the regions is still quite slow and so a different group, Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Son, proposed the faster R-CNN algorithm, which uses a convolutional neural network instead of one of the more traditional segmentation algorithms to propose a blob on those regions, and that would end up running quite a bit faster than the fast R-CNN algorithm. Although, I think the faster R-CNN algorithm, most implementations are usually still quit a bit slower than the YOLO algorithm.

So the idea of region proposals has been quite influential in computer vision. The YOLO or the You Only Look Once algorithm that seems to me like a more promising direction for the long term. But that's my personal opinion and not necessary the opinion of the whole computer vision research committee.

## Semantic segmentation using U-Net

You've learned about object recognition, where the goal is to input a picture and figure out what is in the picture, such as is this a cat or not. You've learned about object detection, where the goal is to further put a bounding box around the object is found. 

Now, you learn about a set of algorithms that's even one step more sophisticated, which is semantic segmentation, where the goal is to draw a careful outline around the object that is detected so that you know exactly which pixels belong to the object and which pixels don't. This type of algorithm, semantic segmentation is useful for many commercial applications as well today. Let's dive in. 

### What is semantic segmentation? 

![image](img/ch_21/img_22.jpg)

Let's say you're building a self-driving car and you see an input image like this and you'd like to detect the position of the other cars. If you use an object detection algorithm, the goal may be to draw bounding boxes like these around the other vehicles. This might be good enough for self-driving car, but if you want your learning algorithm to figure out what is every single pixel in this image, then you may use a semantic segmentation algorithm whose goal is to output a mask of the image. 

For example, rather than detecting the road and trying to draw a bounding box around the roads, which isn't going to be that useful, with semantic segmentation the algorithm attempts to label every single pixel as is this drivable roads or not, indicated by the dark green there. One of the uses of semantic segmentation is that it is used by some self-driving car teams to figure out exactly which pixels are safe to drive over because they represent a drivable surface. 

![image](img/ch_21/img_23.jpg)

Let's look at some other applications. These are a couple of images from research papers by Novikov et al and by Dong et al. In medical imaging, given a chest X-ray, you may want to diagnose if someone has a certain condition, but what may be even more helpful to doctors, is if you can segment out in the image, exactly which pixels correspond to certain parts of the patient's anatomy. In the image on the left, the lungs, the heart, and the clavicle, so the collarbones are segmented out using different colors. This segmentation can make it easier to spot irregularities and diagnose serious diseases and also help surgeons with planning out surgeries. 

In this example, a brain MRI scan is used for brain tumor detection. Manually segmenting out this tumor is very time-consuming and laborious, but if a learning algorithm can segment out the tumor automatically; this saves radiologists a lot of time and this is a useful input as well for surgical planning. The algorithm used to generate this result is an algorithm called UNet. 

### What semantic segmentation actually does. 

![image](img/ch_21/img_24.jpg)

For the sake of simplicity, let's use the example of segmenting out a car from some background. Let's say for now that the only thing you care about is segmenting out the car in this image. In that case, you may decide to have two classes labels. One for a car and zero for not car. In this case, the job of the segmentation algorithm of the unit algorithm will be to output, either one or zero for every single pixel in this image, where a pixel should be labeled one, if it is part of the car and label zero if it's not part of the car. 

Alternatively, if you want to segment this image, looking more finely you may decide that you want to label the car one. Maybe you also want to know where the buildings are. In which case you would have a second class, class two the building and then finally the ground or the roads, class three, in which case the job the learning algorithm would be to label every pixel as follows instead. Taking the per-pixel labels and shifting it to the right, this is the output that we would like to train a UNet. This is a lot of outputs, instead of just giving a single class label or maybe a class label and coordinates needed to specify bounding box the neural network unit in this case, has to generate a whole matrix of labels. 

### What's the right neural network architecture to do that? 

![image](img/ch_21/img_25.jpg)

Let's start with the object recognition neural network architecture that you're familiar with and let's figure how to modify this in order to make this new network output, a whole matrix of class labels. Here's a familiar convolutional neural network architecture, where you input an image which is fed forward through multiple layers in order to generate a class label y hat. In order to change this architecture into a semantic segmentation architecture, let's get rid of the last few layers and one key step of semantic segmentation is that, whereas the dimensions of the image have been generally getting smaller as we go from left to right, it now needs to get bigger so they can gradually blow it back up to a full-size image, which is a size you want for the output. 

Specifically, this is what a UNet architecture looks like. As we go deeper into the unit, the height and width will go back up while the number of channels will decrease so the unit architecture looks like this until eventually, you get your segmentation map of the cat. One operation we have not yet covered is the operation that makes the image bigger. To explain how that works, you have to know how to implement a transpose convolution. 

That's semantic segmentation, a very useful algorithm for many computer vision applications where the key idea is you have to take every single pixel and label every single pixel individually with the appropriate class label. As you've seen in this video, a key step to do that is to take a small set of activations and to blow it up to a bigger set of activations. In order to do that, you have to implement something called the transpose convolution, which is important operation that is used multiple times in the UNet architecture.

## Transpose Convolutions

The transpose convolution is a key part of the UNet architecture. How do you take a two-by-two inputs and blow it up into a four- by-four-dimensional output? 

![image](img/ch_21/img_26.jpg)

The transpose convolution lets do that. Let's dig into the details. 

You're familiar with the normal convolution in which a typical layer of a new network may input a 6 by 6 by 3 image, convolve that with a set of, say, 3 by 3 by 3 filters and if you have 5 of these, then you end up with an output that is 4 by 4 by 5. 

A transpose convolution looks a bit different. You might inputs a 2 by 2, set of activation, convolve that with a 3 by 3 filter, and end up with an output that is 4 by 4, that's bigger than the original inputs. Let's step through a more detailed example of how this works. 

![image](img/ch_21/img_27a.jpg)

In this example, we're going to take a 2 by 2 input like they're shown on the left and we want to end up with a four by four output. But to go from two-by-two to four-by-four, I'm going to choose to use a filter that is f by f, and I'm going to choose 3 by 3. I'm also going to use a padding p equal to 1 and I'm going to use a stride s equal to 2 for this example. 

Let's see how the transpose convolution will work. In the regular convolution, you would take the filter and place it on top of the inputs and then multiply and sum up. In the transpose convolution, instead of placing the filter on the input, you would instead place a filter on the output. 

Let me illustrate that by mechanically stepping through the steps of the transpose convolution calculation. Let's starts with this upper left entry of the input, which is a two. We are going to take this number 2 and multiply it by every single value in the filter and we're going to take the output which is 3 by 3 and paste it in this position. Now, the padding area isn't going to contain any values. What we're going to end up doing is ignore this padded region and just throw in four values in the red highlighted area: specifically, the upper left entry is 0 times 2, so that's 0. The second entry is 1 times 2, that is 2. Down here is 2 times 2, that's 4, and then over here is 1 times 2 so that's equal to 2. 

![image](img/ch_21/img_27b.jpg)

Next, let's look at the second entry of the input which is a 1. I'm going to switch to green pen for this. Once again, we're going to take a 1 and multiply by 1 every single elements of the filter, because we're using a stride of 2, we're now going to shift to box in which we copy the numbers over by two steps. Again, we'll ignore the area which is in the padding and the only area that we need to copy the numbers over is this green shaded area. You may notice that there is some overlap between the places where we copy the red-colored version of the filter and the green version and we cannot simply copy the green value over the red one. Where the red and the green boxes overlap, you add two values together. 

Where there's already a 2 from the first weighted filter, you add to it this first value from the green region which is also 2. You end up with 2 plus 2. The next entry, 0 times 1 is 0, then you have 1, 2 plus 0 times 1, so 2 plus 0 followed by 2, followed by 1 and again, we shifted 2 squares over from the red box here to the green box here because it using a stride of two. 

![image](img/ch_21/img_27c.jpg)

Next, let's look at the lower-left entry of the input, which is 3. We'll take the filter, multiply every element by 3 and we've gone down by one step here. We're going to go down by two steps here. We will be filling in our numbers in this three by three square and you find that the numbers you copying over are 2 times 3, which is 6, 1 times 3, which is 3, 0 times 3, which is 0, and so on, 3, 6, 3. 

![image](img/ch_21/img_27d.jpg)

Then lastly, let's go into the last input element, which is 2. We will multiply every elements of the filter by 2 and add them to this block and you end up with adding 1 times 2 which is plus 2, and so on for the rest of the elements. 

The final step is to take all the numbers in these four by four matrix of values in the 16 values and add them up, hence that's your four-by-four outputs. 

### Why do we have to do it this way?

In case you're wondering why do we have to do it this way, I think there are multiple possible ways to take small inputs and turn it into bigger outputs, but the Transpose Convolution happens to be one that is effective and when you learn all the parameters of the filter here. This turns out to give good results when you put this in the context of the UNet which is the learning algorithm will use now. 

We step through step-by-step how the transpose convolution lets you take a small input, say 2 by 2, and blow it up into larger output, say 4 by 4. Now that you understand a Transpose Convolution, let's take this building block you now have and see how it fits into the overall UNet architecture that you can use for semantic segmentation.

## U-Net Architecture intuition

Armed of the Transpose Convolution, you're now ready to dive into the details of the UNet architecture. Let's first go over the architecture quickly to build intuition about how the UNet works. And in the next section, we'll go through the details together. 

![image](img/ch_21/img_28.jpg)

Here's a rough diagram of the neural network architecture for semantic segmentation. And so we use normal convolutions for the first part of the neural network, similarly to the earlier neural networks that you've seen. 

The first part of the neural network will sort of compress the image. You've gone from a very large image to one where the height end the width of this activation is much smaller. So you've lost a lot of spatial information because the dimension is much smaller, but it's much deeper. So, for example, this middle layer may represent if there's a cat in the lower right hand portion of the image. But the detailed spatial information is lost because of height end width is much smaller. 

Then the second half of this neural network uses the transports convolution to blow the representation size up back to the size of the original input image. 

Now it turns out that there's one modification to this architecture that would make it work much better. And that's what we turn this into the UNet architecture, which is that skip connections from the earlier layers to the later layers like this. So that this earlier block of activations is copied directly to this later block. So why do we want to do this? It turns out that for this, next final layer is to decide which region is a cat. 

If we consider the last layer of the UNet, two types of information are useful:
1. One is the high level, spatial, high level contextual information which it gets from the previous layer. Where hopefully the neural network, have figured out that in the lower right corner of the image or maybe in the right part of the image, there's some cat. But this previous layer has lower spatial resolution: to height and width with is just low. 
2. And a very detailed, fine grained spatial information. So what the skip connection does is it allows the neural network to take the input layer very high resolution and low level feature information where it could capture, for every pixel position, how much fairy stuff is there in this pixel, and used the skip connection to pass that directly to this later layer. 

And so the last layer has both the lower resolution, but high level, spatial, contextual information, as well as the low level detailed texture information in order to make a decision as to whether a certain pixel is part of a cat or not. 

## U-Net Architecture

I found the UNet architecture useful for many applications, and it's one of the most important and fundamental neural network architectures of computer vision today. Le's see how the U-Net works. 

![image](img/ch_21/img_29.jpg)

This is what a U-Net looks like. It's called a UNet, because when you draw it like this, it looks a lot like a U. These ideas were due to Olaf Ranneberger, Philipp Fisher and Thomas Bronx. When they wrote the original UNet paper, they were thinking on the application of biomedical image segmentation: segmenting medical images. But these ideas turned out to be useful for many other computer vision, semantic segmentation applications as well. 

So the input to the UNet is an image, H by W by 3, for three RGB channels. 

I'm going to visualize this image as a thin layer like that. I know that previously we had taken neural network layers and drawn them, as 3D blocks, but in order to make the UNet diagram look simpler we are going to look just at the edge of it from now on: so all you see is the edge of a solid, and the rest is hidden behind this dark blue rectangle. 

The height of the layer is h and the width of the edge is 3, and represents the number of channels, the depth is w. 

Now, the first part of the UNet uses normal feed forward neural network convolutional layers. So I'm going to use a black arrow to denote a convolutional layer, followed by a ReLU activation function. Then in the next layer, we have increased the number of channels a little bit, but the dimension is still height by width w and then another convolutional layer with another activation function.

Now we're still in the first half of the neural network. We're going to use Max pooling to reduce the height and width. 

Then you end up with a set of activations where the height h and width w is lower, but the number of channels is increasing. Then we have two more layers of normal feed forward convolutions with a ReLU activation function, and then we supply Max pooling again. And then you repeat again. So far this is the normal convolution layers with activation functions that you've been used to.

So notice that the height of this layer I deal with is now very small. So we're going to start to apply transpose convolution layers, which I'm going to note by the green arrow in order to build the dimension of this neural network back up. So with the first transpose convolutional layer or trans conv layer, you're going to get a set of activations that looks like that: in the first trans conv layers, we did not increase the height and width, but we did decrease the number of channels. And there's one more thing you need to do to build a UNet, which is to add in that skip connection which I'm going to denote with this grey arrow. What the skip connection does is it takes this set of activations and just copies it over to the right. And so the set of activations you end up with is like this. The light blue part comes from the transpose convolution, and the dark blue part is just copied over from the left. 

To keep on building up the UNet we are going to apply a couple more layers of the regular convolutions, followed by our ReLU activation function so denoted by the black arrows, and then we apply another transpose convolutional layer. So green arrow and here we're going to start to increase the dimensions: the height and width of the output image. And so now the height is getting bigger. And again we're going to apply a skip connection. So there's a grey arrow again where they take this set of activations on the left and just copy it over to the right.

We have then more convolutional layers and other transpose convolution, skip connection. Once again, we're going to take this set of activations and copy it over to the right and then more convolutional layers, followed by another transpose convolution. Skip connection, copy that over. And now we're back to a set of activations that is the original input images, height and width. We're going to have a couple more layers of a normal fee forward convolutions, and then finally, to take this and map this to our segmentation map, we're going to use a 1 by 1 convolution which I'm going to denote with that magenta arrow to finally give us this which is going to be our output.

The dimensions of this output layer is going to be h by w, so the same dimensions as our original input by the number of classes. So if you have three classes to try and recognize, this will be three. If you have ten different classes to try to recognize in your segmentation at then that last number will be ten. 

### FInal step, very important

And so what this does is: for everyone of your class you have h by w pixels that tell (with a probability 0 or 1) for each pixel how likely is that pixel to come from each of these different classes. And if you take a arg max over these n classes, then that's how you classify each of the pixels into one of the classes, and you can visualize it like the segmentation map showing on the right. So that's it. 

You've learned about the transpose convolution and the UNet architecture.

## Addendum

### Leaky ReLU

Leaky ReLU (Rectified Linear Unit) is a variant of the ReLU activation function commonly used in neural networks. It is defined as:

$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{otherwise}
\end{cases}
$

where α is a small constant, typically around 0.01. Unlike the standard ReLU function, which outputs 0 for any negative input, Leaky ReLU allows a small, non-zero gradient for negative inputs, which helps mitigate the issue of "dying ReLU" where neurons get stuck in the negative side and cease to learn.

The choice of α is usually made based on empirical observations and can be tuned during model training. Leaky ReLU has been shown to perform well in many neural network architectures and is commonly used as an alternative to ReLU, especially when dealing with vanishing gradients.

### Tensorflow why I can use sequential model if I need to ad skip connections?

In TensorFlow, the Sequential model is a linear stack of layers, where each layer has exactly one input tensor and one output tensor. This makes it suitable for building simple feedforward neural networks where each layer feeds its output to the next layer in sequence.

However, when you need to add skip connections or implement more complex network architectures like residual networks (ResNet), you cannot use the Sequential model directly because it does not support merging outputs from multiple layers or adding connections that skip over intermediate layers.

To implement skip connections or more complex architectures, you would typically use the Functional API or the Subclassing API in TensorFlow.

Functional API: With the Functional API, you can define a model by explicitly connecting layers to form a directed acyclic graph (DAG). This allows you to create complex network architectures with multiple inputs, multiple outputs, skip connections, shared layers, etc. You can use the tf.keras.Model class to create models with the Functional API.

Subclassing API: With the Subclassing API, you can define a custom model class by subclassing the tf.keras.Model class and implementing the call method. This gives you maximum flexibility to define the forward pass of your model using TensorFlow operations directly. You can create custom layers, loops, conditionals, and any other computation you need within the call method.

In summary, if you need to add skip connections or implement complex architectures, you should use the Functional API or the Subclassing API instead of the Sequential model.
