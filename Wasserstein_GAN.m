%% 
% Train Wasserstein GAN with Gradient Penalty (WGAN-GP)
% To optimize the performance of the generator, maximize the loss of the discriminator when given generated data.
% That is, the objective of the generator is to generate data that the discriminator classifies as "real". To optimize t
% performance of the discriminator, minimize the loss of the discriminator when given batches of both real and
% generated data. That is, the objective of the discriminator is to not be "fooled" by the generator.
% Ideally, these strategies result in a generator that generates convincingly realistic data and a discriminator that has
% learned strong feature representations that are characteristic of the training data. However, [2] argues that the
% divergences which GANs typically minimize are potentially not continuous with respect to the generator’s
% parameters, leading to training difficulty and introduces the Wasserstein GAN (WGAN) model that uses 
% Wasserstein loss to help stabilize training. A WGAN model can still produce poor samples or fail to converge
% because interactions between the weight constraint and the cost function can result in vanishing or exploding
% gradients. To address these issues, [3] introduces a gradient penalty which improves stability by penalizing
% gradients with large norm values at the cost of longer computational time. This type of model is known as a
% WGAN-GP model.
% This example shows how to train a WGAN-GP model that can generate signals with similar characteristics to a
% training set of images.

downloadFolder = "sig/";
filename = fullfile("sig/", "sigs.PNG");
imageFolder = fullfile(downloadFolder,"sigs.PNG");
datasetFolder = fullfile(imageFolder);
imds = imageDatastore(datasetFolder, ...
 IncludeSubfolders=true);

netD = dlnetwork; 

augmenter = imageDataAugmenter(RandXReflection=true);
augimds = augmentedImageDatastore([64 64],imds,DataAugmentation=augmenter);

netD = dlnetwork;

numFilters = 64;
scale = 0.2;
inputSize = [64 64 3];
filterSize = 5;
layersD = [
 imageInputLayer(inputSize,Normalization="none")
 convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
 leakyReluLayer(scale)
 convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
 layerNormalizationLayer
 leakyReluLayer(scale)
 convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
 layerNormalizationLayer
 leakyReluLayer(scale)
 convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
 layerNormalizationLayer
 leakyReluLayer(scale)
 convolution2dLayer(4,1)
 sigmoidLayer];
netD = addLayers(netD, layersD);


netD = initialize(netD);
netG = dlnetwork; 
filterSize = 5;
numFilters = 64;
numLatentInputs = 100;
projectionSize = [4 4 512];
layersG = [
 featureInputLayer(numLatentInputs,Normalization="none")
 transposedConv2dLayer(filterSize,4*numFilters)
 reluLayer
 transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
 reluLayer
 transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
 reluLayer
 transposedConv2dLayer(filterSize,3,Stride=2,Cropping="same")
 tanhLayer];
netG = addLayers(netG, layersG);
netG = initialize(netG);
miniBatchSize = 64;
numIterationsG = 10000;
numIterationsDPerG = 5;
lambda = 10;
learnRateD = 2e-4;
learnRateG = 1e-3;
gradientDecayFactor = 0;
squaredGradientDecayFactor = 0.9;
validationFrequency = 20; 
augimds.MiniBatchSize = miniBatchSize;
mbq = minibatchqueue(augimds,...
 MiniBatchSize=miniBatchSize,...
 PartialMiniBatch="discard",...
 MiniBatchFcn=@preprocessMiniBatch,...
 MiniBatchFormat="SSCB");
trailingAvgD = [];
trailingAvgSqD = [];
trailingAvgG = [];
trailingAvgSqG = [];
numValidationImages = 25;
ZValidation = randn(numLatentInputs,numValidationImages,"single");
ZValidation = dlarray(ZValidation,"CB");
if canUseGPU
 ZValidation = gpuArray(ZValidation);
end

iterationG = 0;
iterationD = 0;
start = tic;
% Loop over mini-batches
while iterationG < numIterationsG
 iterationG = iterationG + 1;
 % Train discriminator only
 for n = 1:numIterationsDPerG
 iterationD = iterationD + 1;
 % Reset and shuffle mini-batch queue when there is no more data.
 if ~hasdata(mbq)
 shuffle(mbq);
 end
 % Read mini-batch of data.
 X = next(mbq);
 % Generate latent inputs for the generator network. Convert to
 % dlarray and specify the dimension labels "CB" (channel, batch).
 Z = randn([numLatentInputs size(X,4)],like=X);
 Z = dlarray(Z,"CB");
 % Evaluate the discriminator model loss and gradients using dlfeva
 % modelLossD function listed at the end of the example.
 [lossD, lossDUnregularized, gradientsD] = dlfeval(@modelLossD, netD, netG, X, Z, lambda);
 % Update the discriminator network parameters.
 [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, trailingAvgD, trailingAvgSqD, iterationD,  learnRateD, gradientDecayFactor, squaredGradientDecayFactor);
 end
 % Generate latent inputs for the generator network. Convert to dlarray
 % and specify the dimension labels "CB" (channel, batch).
 Z = randn([numLatentInputs size(X,4)],like=X);
 Z = dlarray(Z,"CB");
 % Evaluate the generator model loss and gradients using dlfeval and th
 % modelLossG function listed at the end of the example.
 [~,gradientsG] = dlfeval(@modelLossG, netG, netD, Z);
 % Update the generator network parameters.
 [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, ...
 trailingAvgG, trailingAvgSqG, iterationG, ...
 learnRateG, gradientDecayFactor, squaredGradientDecayFactor);
 % Every validationFrequency generator iterations, display batch of
 % generated images using the held-out generator input
 if mod(iterationG,validationFrequency) == 0 || iterationG == 1
% Generate images using the held-out generator input.
 XGeneratedValidation = predict(netG,ZValidation);
 % Tile and rescale the images in the range [0 1].
 I = imtile(extractdata(XGeneratedValidation));
 I = rescale(I);
 % Display the images.
 subplot(1,2,1);
 % image(imageAxes,I)
 xticklabels([]);
 yticklabels([]);
 title("Generated signal");
 end
 % Update the scores plot
 subplot(1,2,2)
 lossD = double(lossD);
 lossDUnregularized = double(lossDUnregularized);
 addpoints(lineLossD,iterationG,lossD);
 addpoints(lineLossDUnregularized,iterationG,lossDUnregularized);
 D = duration(0,0,toc(start),Format="hh:mm:ss");
 title( ...
 "Iteration: " + iterationG + ", " + ...
 "Elapsed: " + string(D))
 drawnow
end
ZNew = randn(numLatentInputs,25,"single");
ZNew = dlarray(ZNew,"CB");
if canUseGPU
 ZNew = gpuArray(ZNew);
end
XGeneratedNew = predict(netG,ZNew);
I = imtile(extractdata(XGeneratedNew));
I = rescale(I);
figure
image(I)
axis off
title("Generated Images")
function [lossD, lossDUnregularized, gradientsD] = modelLossD(netD, netG, X,Z,lambda)
% Calculate the predictions for real data with the discriminator network.
YPred = forward(netD, X);
% Calculate the predictions for generated data with the discriminator
% network.
XGenerated = forward(netG,Z);
YPredGenerated = forward(netD, XGenerated);
% Calculate the loss.
lossDUnregularized = mean(YPredGenerated - YPred);
% Calculate and add the gradient penalty. 
epsilon = rand([1 1 1 size(X,4)],like=X);
XHat = epsilon.*X + (1-epsilon).*XGenerated;
YHat = forward(netD, XHat);
% Calculate gradients. To enable computing higher-order derivatives, set
% EnableHigherDerivatives to true.
gradientsHat = dlgradient(sum(YHat),XHat,EnableHigherDerivatives=true);
gradientsHatNorm = sqrt(sum(gradientsHat.^2,1:3) + 1e-10);
gradientPenalty = lambda.*mean((gradientsHatNorm - 1).^2);
% Penalize loss.
lossD = lossDUnregularized + gradientPenalty;
% Calculate the gradients of the penalized loss with respect to the
% learnable parameters.
gradientsD = dlgradient(lossD, netD.Learnables);
end
function [lossG,gradientsG] = modelLossG(netG, netD, Z)
% Calculate the predictions for generated data with the discriminator
% network.
XGenerated = forward(netG,Z);
YPredGenerated = forward(netD, XGenerated);
% Calculate the loss.
lossG = -mean(YPredGenerated);
% Calculate the gradients of the loss with respect to the learnable
% parameters.
gradientsG = dlgradient(lossG, netG.Learnables);
end

function X = preprocessMiniBatch(data)
% Concatenate mini-batch
X = cat(4,data{:});
% Rescale the images in the range [-1 1].
X = rescale(X,-1,1,InputMin=0,InputMax=255);
end
