# Classic Networks

In this chapter, we'll see some of the classic neural network architecture starting with LeNet-5, and then AlexNet, and then VGGNet.

## LeNet-5 architecture

![image](ch_20\img_1.jpg)

Here is the LeNet-5 architecture. You start off with an image which say, 32 by 32 by 1. And the goal of LeNet-5 was to recognize handwritten digits, so maybe an image of a digits like that. And LeNet-5 was trained on grayscale images, which is why it's 32 by 32 by 1. This neural network architecture is actually quite similar to the last example you saw last week. In the first step, you use a set of six, 5 by 5 filters with a stride of one because you use six filters you end up with a 20 by 20 by 6 over there. And with a stride of one and no padding, the image dimensions reduces from 32 by 32 down to 28 by 28. Then the LeNet neural network applies pooling. And back then when this paper was written, people use average pooling much more. If you're building a modern variant, you probably use max pooling instead. But in this example, you average pool and with a filter width two and a stride of two, you wind up reducing the dimensions, the height and width by a factor of two, so we now end up with a 14 by 14 by 6 volume. I guess the height and width of these volumes aren't entirely drawn to scale. Now technically, if I were drawing these volumes to scale, the height and width would be stronger by a factor of two. Next, you apply another convolutional layer. This time you use a set of 16 filters, the 5 by 5, so you end up with 16 channels to the next volume. And back when this paper was written in 1998, people didn't really use padding or you always using valid convolutions, which is why every time you apply convolutional layer, they heightened with strengths. So that's why, here, you go from 14 to 14 down to 10 by 10. Then another pooling layer, so that reduces the height and width by a factor of two, then you end up with 5 by 5 over here. And if you multiply all these numbers 5 by 5 by 16, this multiplies up to 400. That's 25 times 16 is 400. And the next layer is then a fully connected layer that fully connects each of these 400 nodes with every one of 120 neurons, so there's a fully connected layer. And sometimes, that would draw out exclusively a layer with 400 nodes, I'm skipping that here. There's a fully connected layer and then another a fully connected layer. And then the final step is it uses these essentially 84 features and uses it with one final output. I guess you could draw one more node here to make a prediction for ŷ. And ŷ took on 10 possible values corresponding to recognising each of the digits from 0 to 9. A modern version of this neural network, we'll use a softmax layer with a 10 way classification output. Although back then, LeNet-5 actually use a different classifier at the output layer, one that's useless today. So this neural network was small by modern standards, had about 60,000 parameters. And today, you often see neural networks with anywhere from 10 million to 100 million parameters, and it's not unusual to see networks that are literally about a thousand times bigger than this network. But one thing you do see is that as you go deeper in a network, so as you go from left to right, the height and width tend to go down. So you went from 32 by 32, to 28 to 14, to 10 to 5, whereas the number of channels does increase. It goes from 1 to 6 to 16 as you go deeper into the layers of the network. One other pattern you see in this neural network that's still often repeated today is that you might have some one or more conu layers followed by pooling layer, and then one or sometimes more than one conu layer followed by a pooling layer, and then some fully connected layers and then the outputs. So this type of arrangement of layers is quite common. 

### Paper details

Now finally, this is maybe only for those of you that want to try reading the paper. There are a couple other things that were different. The rest of this slide, I'm going to make a few more advanced comments, only for those of you that want to try to read this classic paper. And so, everything I'm going to write in red, you can safely skip on the slide, and there's maybe an interesting historical footnote that is okay if you don't follow fully. So it turns out that if you read the original paper, back then, people used sigmoid and tanh nonlinearities, and people weren't using value nonlinearities back then. So if you look at the paper, you see sigmoid and tanh referred to. And there are also some funny ways about this network was wired that is funny by modern standards. So for example, you've seen how if you have a nh by nw by nc network with nc channels then you use f by f by nc dimensional filter, where everything looks at every one of these channels. But back then, computers were much slower. And so to save on computation as well as some parameters, the original LeNet-5 had some crazy complicated way where different filters would look at different channels of the input block. And so the paper talks about those details, but the more modern implementation wouldn't have that type of complexity these days. And then one last thing that was done back then I guess but isn't really done right now is that the original LeNet-5 had a non-linearity after pooling, and I think it actually uses sigmoid non-linearity after the pooling layer. So if you do read this paper, and this is one of the harder ones to read than the ones we'll go over in the next few videos, the next one might be an easy one to start with. Most of the ideas on the slide I just tried in sections two and three of the paper, and later sections of the paper talked about some other ideas. It talked about something called the graph transformer network, which isn't widely used today. So if you do try to read this paper, I recommend focusing really on section two which talks about this architecture, and maybe take a quick look at section three which has a bunch of experiments and results, which is pretty interesting. 

## AlexNet

![image](ch_20\img_2.jpg)

The second example of a neural network I want to show you is AlexNet, named after Alex Krizhevsky, who was the first author of the paper describing this work. The other author's were Ilya Sutskever and Geoffrey Hinton. So, AlexNet input starts with 227 by 227 by 3 images. And if you read the paper, the paper refers to 224 by 224 by 3 images. But if you look at the numbers, I think that the numbers make sense only of actually 227 by 227. And then the first layer applies a set of 96 11 by 11 filters with a stride of four. And because it uses a large stride of four, the dimensions shrinks to 55 by 55. So roughly, going down by a factor of 4 because of a large stride. And then it applies max pooling with a 3 by 3 filter. So f equals three and a stride of two. So this reduces the volume to 27 by 27 by 96, and then it performs a 5 by 5 same convolution, same padding, so you end up with 27 by 27 by 276. Max pooling again, this reduces the height and width to 13. And then another same convolution, so same padding. So it's 13 by 13 by now 384 filters. And then 3 by 3, same convolution again, gives you that. Then 3 by 3, same convolution, gives you that. Then max pool, brings it down to 6 by 6 by 256. If you multiply all these numbers,6 times 6 times 256, that's 9216. So we're going to unroll this into 9216 nodes. And then finally, it has a few fully connected layers. And then finally, it uses a softmax to output which one of 1000 causes the object could be. So this neural network actually had a lot of similarities to LeNet, but it was much bigger. So whereas the LeNet-5 from previous slide had about 60,000 parameters, this AlexNet that had about 60 million parameters. And the fact that they could take pretty similar basic building blocks that have a lot more hidden units and training on a lot more data, they trained on the image that dataset that allowed it to have a just remarkable performance. Another aspect of this architecture that made it much better than LeNet was using the ReLU activation function. 

### Paper details

And then again, just if you read the bay paper some more advanced details that you don't really need to worry about if you don't read the paper, one is that, when this paper was written, GPUs was still a little bit slower, so it had a complicated way of training on two GPUs. And the basic idea was that, a lot of these layers was actually split across two different GPUs and there was a thoughtful way for when the two GPUs would communicate with each other. And the paper also, the original AlexNet architecture also had another set of a layer called a Local Response Normalization. And this type of layer isn't really used much, which is why I didn't talk about it. But the basic idea of Local Response Normalization is, if you look at one of these blocks, one of these volumes that we have on top, let's say for the sake of argument, this one, 13 by 13 by 256, what Local Response Normalization, (LRN) does, is you look at one position. So one position height and width, and look down this across all the channels, look at all 256 numbers and normalize them. And the motivation for this Local Response Normalization was that for each position in this 13 by 13 image, maybe you don't want too many neurons with a very high activation. But subsequently, many researchers have found that this doesn't help that much so this is one of those ideas I guess I'm drawing in red because it's less important for you to understand this one. And in practice, I don't really use local response normalizations really in the networks language trained today. 

### Paper considerations

So if you are interested in the history of deep learning, I think even before AlexNet, deep learning was starting to gain traction in speech recognition and a few other areas, but it was really just paper that convinced a lot of the computer vision community to take a serious look at deep learning to convince them that deep learning really works in computer vision. And then it grew on to have a huge impact not just in computer vision but beyond computer vision as well. And if you want to try reading some of these papers yourself and you really don't have to for this course, but if you want to try reading some of these papers, this one is one of the easier ones to read so this might be a good one to take a look at. So whereas AlexNet had a relatively complicated architecture, there's just a lot of hyperparameters, right? Where you have all these numbers that Alex Krizhevsky and his co-authors had to come up with. 

## VGG-16 network

![image](ch_20\img_3.jpg)

![image](ch_20\img_4.jpg)

Let me show you a third and final example on this video called the VGG or VGG-16 network. And a remarkable thing about the VGG-16 net is that they said, instead of having so many hyperparameters, let's use a much simpler network where you focus on just having conv-layers that are just three-by-three filters with a stride of one and always use same padding. And make all your max pulling layers two-by-two with a stride of two. And so, one very nice thing about the VGG network was it really simplified this neural network architectures. So, let's go through the architecture. So, you solve up with an image for them and then the first two layers are convolutions, which are therefore these three-by-three filters. And in the first two layers use 64 filters. You end up with a 224 by 224 because using same convolutions and then with 64 channels. So because VGG-16 is a relatively deep network, am going to not draw all the volumes here. So what this little picture denotes is what we would previously have drawn as this 224 by 224 by 3. And then a convolution that results in I guess a 224 by 224 by 64 is to be drawn as a deeper volume, and then another layer that results in 224 by 224 by 64. So this conv64 times two represents that you're doing two conv-layers with 64 filters. And as I mentioned earlier, the filters are always three-by-three with a stride of one and they are always same convolutions. So rather than drawing all these volumes, am just going to use text to represent this network. Next, then uses are pulling layer, so the pulling layer will reduce. I think it goes from 224 by 224 down to what? Right. Goes to 112 by 112 by 64. And then it has a couple more conv-layers. So this means it has 128 filters and because these are the same convolutions, let's see what is the new dimension. Right? It will be 112 by 112 by 128 and then pulling layer so you can figure out what's the new dimension of that. And now, three conv-layers with 256 filters to the pulling layer and then a few more conv-layers, pulling layer, more conv-layers, pulling layer. And then it takes this final 7 by 7 by 512 these in to fully connected layer, fully connected with four thousand ninety six units and then a softmax output one of a thousand classes. By the way, the 16 in the VGG-16 refers to the fact that this has 16 layers that have weights. And this is a pretty large network, this network has a total of about 138 million parameters. And that's pretty large even by modern standards. But the simplicity of the VGG-16 architecture made it quite appealing. You can tell his architecture is really quite uniform. There is a few conv-layers followed by a pulling layer, which reduces the height and width, right? So the pulling layers reduce the height and width. You have a few of them here. But then also, if you look at the number of filters in the conv-layers, here you have 64 filters and then you double to 128 double to 256 doubles to 512. And then I guess the authors thought 512 was big enough and did double on the game here. But this sort of roughly doubling on every step, or doubling through every stack of conv-layers was another simple principle used to design the architecture of this network. And so I think the relative uniformity of this architecture made it quite attractive to researchers. The main downside was that it was a pretty large network in terms of the number of parameters you had to train. And if you read the literature, you sometimes see people talk about the VGG-19, that is an even bigger version of this network. And you could see the details in the paper cited at the bottom by Karen Simonyan and Andrew Zisserman. But because VGG-16 does almost as well as VGG-19. A lot of people will use VGG-16. But the thing I liked most about this was that, this made this pattern of how, as you go deeper and height and width goes down, it just goes down by a factor of two each time for the pulling layers whereas the number of channels increases. And here roughly goes up by a factor of two every time you have a new set of conv-layers. So by making the rate at which it goes down and that go up very systematic, I thought this paper was very attractive from that perspective. So that's it for the three classic architecture's. If you want, you should really now read some of these papers. I recommend starting with the AlexNet paper followed by the VGG net paper and then the LeNet paper is a bit harder to read but it is a good classic once you go over that. But next, let's go beyond these classic networks and look at some even more advanced, even more powerful neural network architectures. Let's go onto the next video.

## Recall: Vanishing and Exploding Gradients in Neural Network

Vanishing and exploding gradients are common issues encountered during the training of deep neural networks, particularly in networks with many layers. These problems can hinder the convergence of the network and make training difficult or even impossible.

### Vanishing Gradients:

Definition: Vanishing gradients occur when the gradients of the loss function with respect to the parameters of the network become extremely small as they are backpropagated through the network layers.

Causes: This typically occurs in deep networks with many layers, especially in recurrent neural networks (RNNs) or networks utilizing activation functions with limited derivative values such as sigmoid or tanh.

Effects: When gradients vanish, it means that the updates to the parameters during training are minimal, leading to very slow learning or a complete halt in learning.

#### Solutions:

Use activation functions with more favorable properties regarding gradient flow, such as ReLU (Rectified Linear Unit) or variants like Leaky ReLU.

Batch normalization can also help mitigate vanishing gradients by normalizing the inputs to each layer.

Residual connections, as seen in ResNet architectures, allow gradients to flow directly through shortcut connections, bypassing potentially vanishing pathways.

### Exploding Gradients:

Definition: Exploding gradients occur when the gradients of the loss function with respect to the parameters of the network become extremely large during training.

Causes: This often happens when the network is initialized with large weights or when gradients are amplified through deep networks during backpropagation.

Effects: Exploding gradients can lead to large updates to the network parameters, causing instability and making the optimization process erratic.

#### Solutions:

Gradient clipping: Limiting the magnitude of gradients during training can prevent them from becoming too large. This involves scaling gradients if they exceed a certain threshold.

Weight initialization techniques: Initializing network weights properly, such as using Xavier or He initialization, can help mitigate exploding gradients by preventing large initial values.

Using gradient normalization techniques like L2 or L1 normalization can help stabilize gradient magnitudes.

Reduce the learning rate: Sometimes, simply reducing the learning rate can help prevent gradient explosions by slowing down the updates to network parameters.

Both vanishing and exploding gradients are critical challenges in training deep neural networks, and addressing them appropriately is essential for achieving successful training and convergence.

## ResNets

![image](ch_20\img_5.jpg)

![image](ch_20\img_6.jpg)

![image](ch_20\img_7.jpg)

Very, very deep neural networks are difficult to train, because of vanishing and exploding gradient types of problems. In this video, you'll learn about skip connections which allows you to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network. And using that, you'll build ResNet which enables you to train very, very deep networks. Sometimes even networks of over 100 layers. Let's take a look. ResNets are built out of something called a residual block, let's first describe what that is. Here are two layers of a neural network where you start off with some activations in layer a[l], then goes a[l+1] and then deactivation two layers later is a[l+2]. So let's to through the steps in this computation you have a[l], and then the first thing you do is you apply this linear operator to it, which is governed by this equation. So you go from a[l] to compute z[l +1] by multiplying by the weight matrix and adding that bias vector. After that, you apply the ReLU nonlinearity, to get a[l+1]. And that's governed by this equation where a[l+1] is g(z[l+1]). Then in the next layer, you apply this linear step again, so is governed by that equation. So this is quite similar to this equation we saw on the left. And then finally, you apply another ReLU operation which is now governed by that equation where G here would be the ReLU nonlinearity. And this gives you a[l+2]. So in other words, for information from a[l] to flow to a[l+2], it needs to go through all of these steps which I'm going to call the main path of this set of layers. In a residual net, we're going to make a change to this. We're going to take a[l], and just first forward it, copy it, match further into the neural network to here, and just at a[l], before applying to non-linearity, the ReLU non-linearity. And I'm going to call this the shortcut. So rather than needing to follow the main path, the information from a[l] can now follow a shortcut to go much deeper into the neural network. And what that means is that this last equation goes away and we instead have that the output a[l+2] is the ReLU non-linearity g applied to z[l+2] as before, but now plus a[l]. So, the addition of this a[l] here, it makes this a residual block. And in pictures, you can also modify this picture on top by drawing this picture shortcut to go here. And we are going to draw it as it going into this second layer here because the short cut is actually added before the ReLU non-linearity. So each of these nodes here, where there applies a linear function and a ReLU. So a[l] is being injected after the linear part but before the ReLU part. And sometimes instead of a term short cut, you also hear the term skip connection, and that refers to a[l] just skipping over a layer or kind of skipping over almost two layers in order to process information deeper into the neural network. So, what the inventors of ResNet, so that'll will be Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. What they found was that using residual blocks allows you to train much deeper neural networks. And the way you build a ResNet is by taking many of these residual blocks, blocks like these, and stacking them together to form a deep network. So, let's look at this network. This is not the residual network, this is called as a plain network. This is the terminology of the ResNet paper. To turn this into a ResNet, what you do is you add all those skip connections although those short like a connections like so. So every two layers ends up with that additional change that we saw on the previous slide to turn each of these into residual block. So this picture shows five residual blocks stacked together, and this is a residual network. And it turns out that if you use your standard optimization algorithm such as a gradient descent or one of the fancier optimization algorithms to the train or plain network. So without all the extra residual, without all the extra short cuts or skip connections I just drew in. Empirically, you find that as you increase the number of layers, the training error will tend to decrease after a while but then they'll tend to go back up. And in theory as you make a neural network deeper, it should only do better and better on the training set. Right. So, the theory, in theory, having a deeper network should only help. But in practice or in reality, having a plain network, so no ResNet, having a plain network that is very deep means that all your optimization algorithm just has a much harder time training. And so, in reality, your training error gets worse if you pick a network that's too deep. But what happens with ResNet is that even as the number of layers gets deeper, you can have the performance of the training error kind of keep on going down. Even if we train a network with over a hundred layers. And then now some people experimenting with networks of over a thousand layers although I don't see that it used much in practice yet. But by taking these activations be it X or these intermediate activations and allowing it to go much deeper in the neural network, this really helps with the vanishing and exploding gradient problems and allows you to train much deeper neural networks without really appreciable loss in performance, and maybe at some point, this will plateau, this will flatten out, and it doesn't help that much deeper and deeper networks. But ResNet is not even effective at helping train very deep networks. So you've now gotten an overview of how ResNets work. And in fact, in this week's programming exercise, you get to implement these ideas and see it work for yourself. But next, I want to share with you better intuition or even more intuition about why ResNets work so well, let's go onto the next video.

## Data Normalization

When I normalize the input data, the input can be +-1 right? The normalization or relative numbers is still a relative numeber?

When you normalize input data, you typically scale it so that it falls within a specific range, often between 0 and 1 or -1 and 1. However, whether the normalized data can range from -1 to 1 specifically depends on the normalization technique you choose and the nature of your data.

Normalization is about scaling the values of features so that they fall within a specific range without distorting differences in the ranges of values. The purpose is to make different features more comparable and to ensure that one feature doesn't dominate due to its larger scale.

For instance, in min-max normalization, you scale the data to a range between 0 and 1:

If you want to normalize to a range of -1 to 1, you can modify this equation as follows:

In standardization (z-score normalization), the data is scaled to have a mean of 0 and a standard deviation of 1. In this case, the normalized values can be both positive and negative, but they won't necessarily fall within the range of -1 to 1.

So, whether the normalized input can be within the range of +-1 depends on the normalizaation technique used and the distribution of your data. However, even if the normalized values fll within that range, they are still relative numbers in the sense that they represent the data's relationship within its own distribution.

![image](ch_20\draft_1.png)

## Why ResNets Work?

![image](ch_20\img_8.jpg)

![image](ch_20\img_9.jpg)

This is about using residual networks with different dimensions

![image](ch_20\img_10.jpg)

## Networks in Networks and 1x1 Convolutions

![image](ch_20\img_11.jpg)

![image](ch_20\img_12.png)

## Inception Network module

![image](ch_20\img_13.jpg)

![image](ch_20\img_14.png)

![image](ch_20\img_15.png)

## Inception Network

![image](ch_20\img_16.png)

![image](ch_20\img_17.png)


