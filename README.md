# VICReg model analysis and downstream applications
## Training the model:
I trained the model on the CIFAR10 dataset (without using the labels)

The overall loss during the training:
![Screenshot 2023-07-18 at 8 42 19](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/343d3f22-d0a3-418c-bbea-59bbaceb1ff2)

The decomposition to the 3 loss components:
![Screenshot 2023-07-18 at 8 42 33](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/8ffe66b0-e167-435d-8635-68e8236e7a86)

The overall loss on the test set:
![Screenshot 2023-07-18 at 8 42 53](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/bb52c098-ec55-4bca-b711-edf1229d9a89)

The decomposition to the 3 loss components:
![Screenshot 2023-07-18 at 8 43 23](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/53f99967-097a-4f4b-89a3-a4c4be1e6394)

## Exploring the representations
The 2D representation of the models embedding with PCA and TSNE decompositions:
![Screenshot 2023-07-18 at 8 44 35](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/b86c8f59-dc94-456d-a8fe-45024e21b2ac)

I think the T-SNE visualization seems more effective for visualization. In both cases it indeed seems that each class is sampled from different distributions on the 2D plane, but in the T-SNE those distributions have much less overlap what makes them in my opinion a better visualization. I think this is due to the fact that the PCA decomposition works globally and tries to preserve the most variance in its features, while the T-SNE has the ability to use local distances, what is more suitable for our embeddings. Looking at the T-SNE visualization we can see that VICReg managed to capture most of the class information accurately, with some classes preforming worse than others.
We can see that the classes that seems most entangled are:
[ horse , deer]
[ dog , cat    ]
This seems unsurprising since the real life semantic of those pairs is very similar, and in the context of the CIFAR10 dataset, with such small images of 32x32, it might be rather hard to differentiate these images correctly using our augmentations.  Other entangles (not as much as the given example) also follow this logic [bird, airplane] [ship, airplane] though this is much les prominent in the visualization of the representation.

The accuracy I got with the linear classifier on the model’s embedding representation, training for 10 epochs, is 62.68 %. This is indeed above the wanted threshold.

## Ablations effects:
### Variance condition ablation:
Training the model with the variance ablation for 30 epochs, I got these PCA T-SNE visualizations plot:
![Screenshot 2023-07-18 at 8 46 45](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/862afe9e-f9ec-4a90-a1eb-ddf83bf77c9e)

The linear prob accuracy is 14.66%
 
We can see the accuracy got much worst. Also, we can see in the visualizations that the classes aren’t as separated as well as in the previous case. I think because we removed the most prominent loss that we saw decreasing during the gradient descend process is the variational loss (also it had large weight µ = 25). The variance loss objective forces each embedding vector in to be different, by making sure each dimension in the representation is meaningful. Without enforcing this condition, the representation might contain similar information in different embedding’s dimensions, what is effectively lowering the actual representation dimension to be much lower, this in tum allows the model to learn much simpler representations, and therefore the results are worst. We can see the simpler representation in the PCA as well from the fact that the data forms lines after the transformation. We can assume that it formed a line in the original space or that it curved on a plane, which gets projected down to a line and not the complex representation of the original model. The TSNE , as we saw in class, calculates the 2D representation with minimization of the KL divergence of the similarity score (as a proxy of the probability function) between all points pairs in the original dimension and the lower dimension.  As we can see the representation it got doesn’t capture the semantic meaning of the representations, what means that the embeddings themselves were of lesser quality than those with the variance loss component.

# Using neighbors instead of augmentations ablation:
The linear probing accuracy is 50.1%
 
As we can see it is lower compered the original linear probing from Q3. 
I think is happens because when training the model without the generated augmentations as neighbors, that were carefully constructed to create a supervision for the model to ignore changes as rotations, crops, and colorations as we said earlier,  and giving instead similar images by the other model’s embeddings distances, we can’t make the same supervision. The neighbors might all have, for say in one class, the same background (let’s say in the ship class, the background is mostly ocean blue), so the model doesn’t learn to ignore the colors in the semantic representation of the image content. This makes the model create inferior embeddings to the original one, and in turn, get worst accuracy in the linear probing. The method doesn’t fail entirely, because the original models embedding were pretty good, so the neighbors are indeed semantically similar in most cases, and mane time look vary different, what still forces the model to learn some of the desired semantic in its embeddings representations.

# Using laplacian eigenmaps ablation (LO):
![Screenshot 2023-07-18 at 8 49 59](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/e98f4ac4-07b1-4078-a8dc-a5a053304887)


It seems that the T-SNE is more effective for downstream object classification, as the classes are separated better what will make the classification simpler.

The Laplacian Eigenmaps, while fitting the embedding, maximize on preserving the wrong information for our semantic tasks. When training the VICReg model, by having the “neighbors” of each image sample to be augmentations of the original image, we essentially teach the model to ignore changes such as rotations, crops, color changes and other non-semantic changes. This forces the model to learn more complex connections between the images.

On the other hand, Laplacian Eigenmaps doesn’t have this kind of supervision when fitting the embeddings. The neighbors are the closest images in the original space where we don’t necessarily have correlation between the similarity of the images to the similarity of the classes, as we can see in this example:

![Screenshot 2023-07-18 at 8 50 36](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/8fda8780-0e36-466e-a6ca-f72e8ff1ac0f)

Because of this, the model is much more prone to learning the “easier” connections between images missing the meaningful semantic attributes that we are interested in. As we can see in the result, this clearly won’t do for a downstream classification task.

## Retrival:

VICReg: nearest images for each class
![Screenshot 2023-07-18 at 8 51 56](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/83baebef-21e8-4fd8-b04a-de5ba3d6e055)

VICReg: most distant images for each class
![Screenshot 2023-07-18 at 8 52 10](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/958394d1-ef9d-4f88-867e-05485e8c5621)

VICReg + neighbors ablation: nearest images for each class
![Screenshot 2023-07-18 at 8 52 33](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/eac66612-59fd-4458-81a4-2f1b9b1672f9)

VICReg + neighbors ablation: most distant images for each class
![Screenshot 2023-07-18 at 8 52 53](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/70057660-2350-4697-a868-46695c5114d9)

It indeed seems that the original VICReg model is able to extract more semantically meaningful relations between the images compared to the model that was trained using the neighbors from the dataset. Most of the images that were the nearest neighbors are from the correct class. I think the “neighbors” model’s representation is more influenced by the colors of the original image, as we can see in the ship class between the models, where the neighbors are also airplanes (white on light blue background), and the original one chose also ships of a different color scheme, and was more correct overall:
![Screenshot 2023-07-18 at 8 53 29](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/cdc3a3b2-e001-4a1c-92b0-4f2249a982e7)

This is also apparent in the car class:
![Screenshot 2023-07-18 at 8 53 54](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/cf15e009-614e-4e1f-a09d-34f515182c3c)

Regarding the most distant images, we can see that they mostly repeat themselves between the classes, in each model. Some of them even appear in all the classes of their respective model. I believe that the representation of them wasn’t captured correctly buy the first model, and later when training the other model, it affected the training.
Overall, I think the original VICReg model performed better in the task of creating a close representation each class and captures the “semantic” as distance between the representations.

# Downstream Applications:

![image](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/c69b4a58-fd9b-4fb9-8d30-3407db2305fe)
As we can see from this ROC plot,  the model without the generated neighbors preforms better at this task, as the AUC score indicates that it separated almost perfectly the two datasets. As we saw before, the model without the neighbors appears to put more emphasis in the embedding on visual similarity (and less on semantical meaning), what may explain why it could separate the images this well as they look distinguishably different from the CIFAR10 dataset images – which contain more colors, textures, and shapes that the humongous MNIST dataset. This is why I believe the model without the generated neighbors performed better detecting the anomalies.

## Anomaly detection (using MNIST as anomalies):
![Screenshot 2023-07-18 at 8 59 49](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/cb0a642f-4f3c-4af1-96f5-76be96f39b5c)
It seems that both models chose similar images as the most anomalous images, an effect that is more pronounced in the model without the augmentations. Because of that I believe that even thou the model without the augmentations separated the anomalies well, it lacks the ordinal information – it classifies the anomalies with the same level of confident as we can see all chosen images are the same. This might be less useful in real life scenario and not this artificially created one, where the semantic meaning, which this model lacks, will probably be more relevant.

## Clustering
![Screenshot 2023-07-18 at 9 00 33](https://github.com/AsafShul/VICReg_selfsupervised_representation_learning/assets/44872433/f6b1e8f8-c13c-4738-b297-953f73bd1504)

It seems to me that the original VICReg model’s clusters are more like the actual classes of the data, as we can clearly see separation in the plots on both coloring methods (by cluster and by class), and we can see sone similarities between those groups. This indicates that we indeed succeeded in the clustering and that our clusters hold semantic data on the classes. In contrast, the coloring based on the class we got in the model without the augmentations seems to be less defined and less correlated to the original classes. I think this indicates that the VICReg model capture the important semantic information regarding the classes better.


