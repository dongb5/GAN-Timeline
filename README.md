# GAN Timeline

This is a timeline showing the development of Generative Adversarial Networks. It is expected to show the evolutions and connections of ideas and to follow the most recent progress in GAN researches.

The paper list partly refers to the lists in [nightrome/really-awesome-gan](https://github.com/nightrome/really-awesome-gan) and [zhangqianhui/AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers).

**Notice:** All dates correspond to the initial version of the submissions.

**Notice:** Papers with **_"Title of this style"_** are the key papers in the development of GANs. Any suggestion about if a paper is key paper or not is welcome!!
***
2014-06-10 | **[Theory]** Ian J. Goodfellow _et al._ **_"Generative Adversarial Networks"_**. **GAN** [arXiv](https://arxiv.org/abs/1406.2661) [code](https://github.com/goodfeli/adversarial)     
- The adversarial nets framework of a generator and a discriminator is first proposed.  
- The framework is a two-player game that the generator is trained to generate images from inputed noises to fool the discriminator while the discriminator is trained to well discriminate real samples and fake samples.  
- The criterion is formulated as `E_real(log(D)) + E_fake(log(1-D))`.         
***
2014-11-06 | **[Theory]** Mehdi Mirza and Simon Osindero. **_"Conditional Generative Adversarial Nets"_**. **CGAN** [arXiv](https://arxiv.org/abs/1411.1784) [code](https://github.com/wiseodd/generative-models)         
- Generative adversarial nets are extended to a conditional model by conditioning both the generator and discriminator on some extra information _y_. _y_ could be any kind of auxiliary information, such as class labels, tags or attributes. the conditioning is performed by feeding _y_ into the both the discriminator and generator as additional input layer.        
***
2015-05-14 | **[Theory]** Gintare Karolina Dziugaite _et al._ **"Training generative neural networks via Maximum Mean Discrepancy optimization"**. [arXiv](https://arxiv.org/abs/1505.03906)                       
***                
2015-06-18 **[Theory]** Emily Denton _et al._ **"Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks"**. **LAPGAN** [arXiv](https://arxiv.org/abs/1506.05751) [code](https://github.com/facebook/eyescream) [blog](http://soumith.ch/eyescream/)     
- The approach uses a cascade of convolutional networks within a Laplacian pyramid framework to generate images in a coarse-to-fine fashion.        
***
2015-11-17 | **[CV App]** Michael Mathieu _et al._ **_"Deep multi-scale video prediction beyond mean square error"_**. [arXiv](https://arxiv.org/abs/1511.05440) [code](https://github.com/dyelax/Adversarial_Video_Generation)         
- LeCun's paper.


2015-11-18 | **[Theory]** Alireza Makhzani _et al._ **_"Adversarial Autoencoders"_**. **AAE** [arXiv](https://arxiv.org/abs/1511.05644)          
- A probabilistic autoencoder that uses the recently proposed generative adversarial networks (GAN) to perform variational inference by matching the aggregated posterior of the hidden code vector of the autoencoder with an arbitrary prior distribution.         
- The paper shows how the adversarial autoencoder can be used in applications such as semi-supervised classification, disentangling style and content of images, unsupervised clustering, dimensionality reduction and data visualization on on MNIST, Street View House Numbers and Toronto Face datasets.         

2015-11-19 | **[Theory]** Alireza Makhzani _et al._ **_"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"_**. **DCGAN** [arXiv](https://arxiv.org/abs/1511.06434) [code](https://github.com/Newmu/dcgan_code) [PytorchCode](https://github.com/pytorch/examples/tree/master/dcgan) [TensorflowCode](https://github.com/carpedm20/DCGAN-tensorflow) [TorchCode](https://github.com/soumith/dcgan.torch) [KerasCode](https://github.com/jacobgil/keras-dcgan)      
- A set of constraints on the architectural topology of Convolutional GANs, name this class of architectures Deep Convolutional GANs (DCGAN), is proposed and evaluated to make them stable to train in most settings.        
- Many interesting visualized samples.                
***          
2016-02-16 | **[Theory]** Daniel Jiwoong Im _et al._ **"Generating images with recurrent adversarial networks"**. **GRAN** [arXiv](https://arxiv.org/abs/1602.05110)                 
- The main difference between `GRAN` versus other generative adversarial models is that the generator _G_ consists of a recurrent feedback loop that takes a sequence of noise samples drawn from the prior distribution _z∼p(z)_ and draws an ouput at multiple time steps.        
- A encoder _g(·)_ and a decoder _f(·)_ are in _G_. At each time step _t_, `C_t = f([z, g(C_t-1)])`. The final generated sample is a mergence of all the outputs of _f(·)_.       
***  
2016-03-12 | **[CV App]** Donggeun Yoo _et al._ **"Pixel-Level Domain Transfer"**. [arXiv(ECCV2016)](https://arxiv.org/abs/1603.07442) [code](https://github.com/fxia22/pldtgan)               


***
2016-05-17 | **[CV App]** Scott Reed _et al._ **"Generative Adversarial Text to Image Synthesis"**. [arXiv](https://arxiv.org/abs/1605.05396) [code](https://github.com/paarthneekhara/text-to-image)               


2016-05-25 | **[Text]** Takeru Miyato _et al._ **"Adversarial Training Methods for Semi-Supervised Text Classification"**. [arXiv](https://arxiv.org/abs/1605.07725)              


2016-05-31 | **[Theory]** Jeff Donahue _et al._ **"Adversarial Feature Learning"**. **BiGANs** [arXiv](https://arxiv.org/abs/1605.09782) [code](https://github.com/wiseodd/generative-models)                 

***  
2016-06-02 | **[Theory]** Vincent Dumoulin _et al._ **"Adversarially Learned Inference"**. **ALI** [arXiv](https://arxiv.org/abs/1606.00704) [code](https://github.com/wiseodd/generative-models)     


2016-06-02 | **[Theory]** Sebastian Nowozin _et al._ **"f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization"**. **f-GAN** [arXiv](https://arxiv.org/abs/1606.00709) [code](https://github.com/wiseodd/generative-models)         
- This paper shows that the generative-adversarial approach is a special case of an existing more general variational divergence estimation approach. It shows that any f-divergence can be used for training generative neural samplers, and _Variational Divergence Minimization (VDM)_ is proposed in this paper.        

2016-06-10 | **[Theory]** Tim Salimans _et al._ **_"Improved Techniques for Training GANs"_**.                  
- Feature matching: Feature matching addresses the instability of GANs by specifying a new objective for the generator. Instead of directly maximizing the output of the discriminator, the new objective requires the generator to generate data that matches the statistics of the real data.            
- Minibatch discrimination:         
- Historical averaging:         
- One-sided label smoothing:         
- Virtual batch normalization:         


2016-06-10 | **[Theory]** Ming-Yu Liu and Oncel Tuzel. **"Coupled Generative Adversarial Networks"**. **CoGAN** [arXiv](https://arxiv.org/abs/1606.07536) [code](https://github.com/wiseodd/generative-models)               
- This work jointly trains two GANs by inputing a signal _Z_ into two typical generator-discriminator architectures and giving two GANs different tasks. During training, the weights of the first few layers of generators and the last few layers of discriminators are shared to learn a joint distribution of images without correspondence supervision.        
- According to the paper, for CV applications, two tasks could be simultaneously generating realistic images and edge images, or normal color images and negative color images, using same input signal.                   
***
2016-09-08 | **[CV app]** Carl Vondrick _et al._ **"Generating Videos with Scene Dynamics"**. [arXiv](https://arxiv.org/abs/1609.02612) [code](https://github.com/cvondrick/videogan) [project](http://web.mit.edu/vondrick/tinyvideo/)                        


2016-09-11 | **[Theory]** Junbo Zhao _et al._ **_"Energy-based Generative Adversarial Network"_**. **EBGAN** [arXiv](https://arxiv.org/abs/1609.03126v1) [code](https://github.com/buriburisuri/ebgan)                        
- The discriminator _D_, whose output is a scalar energy, takes either real or generated images, and estimates the energy value _E_ accordingly.         
- This work chooses to use a margin loss as energy function, but many other choices are possible.        
- This paper devises a regularizaer to ensure the variety of generated images to replace `Minibatch Discrimination (MBD)` since MBD is hard to implemented with energy based discriminators.         

2016-09-12 | **[CV App]** Jun-Yan Zhu _et al._ **_"Generative Visual Manipulation on the Natural Image Manifold"_**. **iGAN** [arXiv](https://arxiv.org/abs/1609.03552) [code](https://github.com/junyanz/iGAN) [project](http://www.eecs.berkeley.edu/%7Ejunyanz/projects/gvm/) [youtube](https://youtu.be/9c4z6YsBGQ0)        


2016-09-18 | **[Theory]** Lantao Yu _et al._ **"SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient"**. **SeqGAN** [arXiv](https://arxiv.org/abs/1609.05473)         

***
2016-10-30 | **[CV App]** Augustus Odena _et al._ **"Conditional Image Synthesis With Auxiliary Classifier GANs"**. [arXiv(ICLR2017)](https://arxiv.org/abs/1610.09585) [code](https://github.com/buriburisuri/ac-gan)     
- _Google Brain_.       

***
2016-11-04 | **[CV App]** Leon Sixt _et al._ **"RenderGAN: Generating Realistic Labeled Data"**. **RenderGAN** [arXiv](https://arxiv.org/abs/1611.01331)                    


2016-11-06 | **[Theory]** Shuangfei Zhai _et al._ **"Generative Adversarial Networks as Variational Training of Energy Based Models"**. **VGAN** [arXiv](https://arxiv.org/abs/1611.01799)            
- This paper proposes `VGAN`, which works by minimizing a variational lower bound of the negative log likelihood (NLL) of an energy based model (EBM), where the model density _p(x)_ is approximated by a variational distribution _q(x)_ that is easy to sample from.            
- It is interesting that two papers of energy-based analysis of GANs are submitted such close.               

2016-11-06 | **[Theory]** Vittal Premachandran and Alan L. Yuille. **"Unsupervised Learning Using Generative Adversarial Training And Clustering]"**. [ICLR2017](https://openreview.net/forum?id=SJ8BZTjeg&noteId=SJ8BZTjeg) [code](https://github.com/VittalP/UnsupGAN)       


2016-11-07 | **[Theory]** Luke Metz _et al._ **"Unrolled Generative Adversarial Networks"**. [arXiv](https://arxiv.org/abs/1611.02163)        


2016-11-07 | **[CV App]** Yaniv Taigman _et al._ **"Unsupervised Cross-Domain Image Generation"**. [arXiv](https://arxiv.org/abs/1611.02200)        


2016-11-07 | **[CV App]** Mickaël Chen and Ludovic Denoyer. **"Multi-view Generative Adversarial Networks"**. [arXiv](https://arxiv.org/abs/1611.02019)        


2016-11-13 | **[Theory]** Xudong Mao _et al._ **_"Least Squares Generative Adversarial Networks"_**. **LSGANs** [arXiv](https://arxiv.org/abs/1611.04076) [code](https://github.com/wiseodd/generative-models)                   
- To overcome the vanishing gradients problem during the learning process, this paper proposes the Least Squares Generative Adversarial Networks (LSGANs) which adopt the least squares loss function for the discriminator.        
- It claims that first, LSGANs are able to generate higher quality images than regular GANs. Second, LSGANs perform more stable during the learning process.        

2016-11-18 | **[Theory]** Xi Chen _et al._ **"InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"**. **InfoGAN** [arXiv](https://arxiv.org/abs/1606.03657) [code](https://github.com/wiseodd/generative-models)      
- This paper describes `InfoGAN`, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation.       
- `InfoGAN` successfully disentangles writing styles from digit shapes on the MNIST dataset, pose from lighting of 3D rendered images, and background digits from the central digit on the SVHN dataset. It also discovers visual concepts that include hair styles, presence/absence of eyeglasses, and emotions on the CelebA face dataset.       

2016-11-18 | **[Theory]** Tarik Arici _et al._ **"Associative Adversarial Networks"**. **AANs** [arXiv](https://arxiv.org/abs/1611.06953)                    


2016-11-19 | **[CV App]** Guim Perarnau _et al._ **"Invertible Conditional GANs for image editing"**. **IcGAN** [arXiv](https://arxiv.org/abs/1611.06355)                    


2016-11-21 | **[CV App]** Phillip Isola _et al._ **_"Image-to-Image Translation with Conditional Adversarial Networks"_**. [arXiv](https://arxiv.org/abs/1611.07004?context=cs) [code](https://github.com/phillipi/pix2pix) [PytorchCode](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [project](https://phillipi.github.io/pix2pix/)       
- Paper from `iGan` research group in UC Berkeley, a development of `iGan`.
- This approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. **Semantic labels → photo**, trained on the Cityscapes dataset. **Architectural labels → photo**, trained on the CMP Facades dataset. **Map → aerial photo**, trained on data scraped from Google Maps. **BW → color photos**. **Edges → photo**, binary edges generated using the HED edge detector. **Sketch → photo**, **Day → night**.        
- The discriminator receives the pairs consisting of sketch and real image as positive samples and the pairs consisting of sketch and fake image as negative samples.                  

2016-11-21 | **[CV App]** Masaki Saito and Eiichi Matsumoto. **"Temporal Generative Adversarial Nets"**. **TGAN** [arXiv](https://arxiv.org/abs/1611.06624)       
- The temporal generator G0 yields a set of latent variables from z0. The image generator G1 transforms them into video frames. The image discriminator D1 first extracts a feature vector from each frame. The temporal discriminator D0 exploits them and evaluates whether these frames are from the dataset or the generator.         


2016-11-25 | **[CV App]** Pauline Luc _et al._ **"Semantic Segmentation using Adversarial Networks"**. [arXiv](https://arxiv.org/abs/1611.08408)       


2016-11-27 | **[CV App]** Arna Ghosh _et al._ **"SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks"**. **SAD-Gan** [arXiv](https://arxiv.org/abs/1611.08788)       


2016-11-29 | **[Music App]** Olof Mogren. **"C-RNN-GAN: Continuous recurrent neural networks with adversarial training"**. **C-RNN-GAN** [arXiv](https://arxiv.org/abs/1611.09904)             
- Both generator and discriminator are built by LSTM-Conv architectures. The generator receives a random variable sequence to generate fake music while the discriminator distinguishes the real and fake music samples.             

2016-11-30 | **[CV App]** Anh Nguyen _et al._ **"Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space"**.[arXiv](https://arxiv.org/abs/1612.00005v1) [code](https://github.com/Evolving-AI-Lab/ppgn)             


***
2016-12-07 | **[Theory]** Tong Che _et al._ **"Mode Regularized Generative Adversarial Networks"**. [arXiv](https://arxiv.org/abs/1612.02136) [code](https://github.com/wiseodd/generative-models)     
- This paper argues that these bad behaviors of GANs are due to the very particular functional shape of the trained discriminators in high dimensional spaces, which can easily make training stuck or push probability mass in the wrong direction, towards that of higher concentration than that of the data generating distribution.        
- It proposes a novel regularizer for the GAN training target. The basic idea is simple yet powerful: in addition to the gradient information provided by the discriminator, the generator is expected to take advantage of other similarity metrics with much more predictable behavior, such asthe L2 norm.                    
- It designs a set of metrics to evaluate the generated samples in terms of both the `diversity of modes` and the distribution fairness of the probability mass. These metrics are shown to be more robust in judging complex generative models, including those which are well-trained and collapsed ones.                

2016-12-10 | **[CV App]** Han Zhang _et al._ **"StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks"**. **StackGAN** [arXiv](https://arxiv.org/abs/1612.03242)      


2016-12-13 | **[Theory]** Daniel Jiwoong Im _et al._ **"Generative Adversarial Parallelization"**. **GAP** [arXiv](https://arxiv.org/abs/1612.04021) [code](https://github.com/wiseodd/generative-models)       
- A framework in which many GANs or their variants are trained simultaneously, exchanging their discriminators, aiming to deal with `missing mode` problem.          
- Every iteration, each generator randomly chooses several discriminators to judge its fake outputs.          

2016-12-13 | **[Theory]** Xun Huang _et al._ **"Stacked Generative Adversarial Networks"**. **SGAN** [arXiv](https://arxiv.org/abs/1612.04357)        
- The model consists of a top-down stack of GANs, each learned to generate lower-level representations conditioned on higher-level representations.        
- A representation discriminator is introduced at each feature hierarchy to encourage the representation manifold of the generator to align with that of the bottom-up discriminative network, leveraging the powerful discriminative representations to guide the generative model.             
- Unlike the original GAN that uses a single noise vector to represent all the variations, `SGAN` decomposes variations into multiple levels and gradually resolves uncertainties in the top-down generative process.                
***     
2017-01-04 | **[CV App]** Junting Pan _et al._ **"SalGAN: Visual Saliency Prediction with Generative Adversarial Networks"**. **SalGAN** [arXiv](https://arxiv.org/abs/1701.01081v2) [project](https://imatge-upc.github.io/saliency-salgan-2017/)      


2017-01-09 | **[Theory]** Ilya Tolstikhin _et al._ **_"AdaGAN: Boosting Generative Models"_**. **AdaGAN** [arXiv](https://arxiv.org/abs/1701.02386)     
- Original GANs are notoriously hard to train and can suffer from the problem of _missing modes (lack of variety)_ where the model is not able to produce examples in certain regions of the space. AdaGan is an iterative procedure where at every step a new component is added into a mixture model by running a GAN algorithm on a reweighted sample. Such an incremental procedure is proved leading to convergence to the true distribution in a finite number of steps if each step is optimal.        


2017-01-17 | **[Theory]** Martin Arjovsky and Léon Bottou. **_"Towards Principled Methods for Training Generative Adversarial Networks"_**. [arXiv](https://arxiv.org/abs/1701.04862)  
- Theory analysis of shortages of original GAN from the authors of `WGAN`.

2017-01-23 | **[Theory]** Guo-Jun Qi **"Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities"**. **LS-GAN** [arXiv](https://arxiv.org/abs/1701.06264) [code](https://github.com/guojunq/lsgan/) **Generalized Loss-Sensitive GAN, GLS-GAN** [code](https://github.com/guojunq/glsgan/)                 
- **Notice!** `Least Squares GANs, LSGANs` and `Loss-Sensitive GANs, LS-GAN`!
- The proposed LS-GAN abandons to learn a discriminator that uses a probability to characterize the likelihood of real samples. Instead, it builds a loss function to distinguish real and generated samples by the assumption that a real example should have a smaller loss than a generated sample.             
- The theoretical analysis presents a regularity condition on the underlying data density, which allows us to use a class of Lipschitz losses and generators to model the LS-GAN. It relaxes the assumption that the classic GAN should have infinite modeling capacity to obtain the similar theoretical guarantee. The paper also proves that the Wasserstein GAN also follows the Lipschitz constraint.        

2017-01-26 | **[Theory]** Martin Arjovsky _et al._ **_"Wasserstein GAN"_**. **WGAN** [arXiv](https://arxiv.org/abs/1701.07875) [code](https://github.com/martinarjovsky/WassersteinGAN)          
- This paper compares several distance measures, namely the _Total Variation_ (TV) distance, the _Kullback-Leibler_ (KL) divergence, the _Jensen-Shannon_ (JS) divergence, and the _Earth-Mover_ (EM, Wasserstein-1) distance, and follows the last one to formulate the criterion.  
- The criterion is formulated as `E_real(D) - E_fake(D)`  
- The paper claim several training tricks, such as weight clipping and using `RMSProp` instead of momentum based methods, like `Adam`.  

2017-01-26 | **[CV App]** Zhedong Zheng _et al._ **"Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro** [arXiv](https://arxiv.org/abs/1701.07717)     
***
2017-02-11 | **[CV App]** Wei Ren Tan _et al._ **"ArtGAN: Artwork Synthesis with Conditional Categorical GANs"**. **ArtGAN** [arXiv](https://arxiv.org/abs/1702.03410)     


2017-02-27 | **[Theory]** R Devon Hjelm _et al._ **_"Boundary-Seeking Generative Adversarial Networks"_**. **BS-GAN** [arXiv](https://arxiv.org/abs/1702.08431) [code](https://github.com/wiseodd/generative-models)     


2017-02-27 | **[CV App]** Zhifei Zhang _et al._ **_"Age Progression/Regression by Conditional Adversarial Autoencoder"_**. [arXiv](https://arxiv.org/abs/1702.08423)       
- This paper proposes a conditional adversarial autoencoder (CAAE) that learns a face manifold, traversing on which smooth age progression and regression can be realized simultaneously. In `CAAE`, the face is first mapped to a latent vector through a convolutional encoder, and then the vector is projected to the face manifold conditional on age through a deconvolutional generator.
***  
2017-03-06 | **[Theory]** Zhiming Zhou _et al._ **"Generative Adversarial Nets with Labeled Data by Activation Maximization"**. **AM-GAN** [arXiv](https://arxiv.org/abs/1703.02000)         
- This paper claims the current GAN model with labeled data still results in undesirable properties due to the overlay of the gradients from multiple classes.              
- It argues that a better gradient should follow the intensity and direction that maximize each sample's activation on one and the only one class in each iteration, rather than weighted-averaging their gradients.        

2017-03-07 | **[Theory]** Chongxuan Li _et al._ **"Triple Generative Adversarial Nets"**. **Triple-GAN** [arXiv](https://arxiv.org/abs/1703.02291)        


2017-03-15 | **[CV App]** Taeksoo Kim _et al._ **"Learning to Discover Cross-Domain Relations with Generative Adversarial Networks"**. **DiscoGAN** [arXiv](https://arxiv.org/abs/1703.05192) [code](https://github.com/wiseodd/generative-models)         
- This work proposes a method based on generative adversarial networks that learns to discover relations between different domains (DiscoGAN).        
- With `DiscoGAN`, many interesting tasks can be done, such as changing hair colors of inputed face images, generating shoes based on inputed bag styles, or generating a car facing the same direction of inputed chair.        

2017-03-17 | **[CV App]** Bo Dai _et al._ **"Towards Diverse and Natural Image Descriptions via a Conditional GAN"**. [arXiv](https://arxiv.org/abs/1703.06029)        


2017-03-23 | **[ML App]** Akshay Mehrotra and Ambedkar Dukkipati. **"Generative Adversarial Residual Pairwise Networks for One Shot Learning"**. [arXiv](https://arxiv.org/abs/1703.08033)         
- This paper uses generated data acts as a strong regularizer for the task of similarity matching and designs a novel network based on th GAN framework that shows improvements for the one shot learning task.        

2017-03-28 | **[Speech]** Santiago Pascual _et al._ **"SEGAN: Speech Enhancement Generative Adversarial Network"**. **SEGAN** [arXiv](https://arxiv.org/abs/1703.09452)          


2017-03-29 | **[CV App]** Jianmin Bao _et al._ **"CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training"**. **CVAE-GAN** [arXiv](https://arxiv.org/abs/1703.10239)          


2017-03-29 | **[CV App]** Kiana Ehsani, _et al._ **"SeGAN: Segmenting and Generating the Invisible"**. **SeGAN** [arXiv](https://arxiv.org/abs/1703.10239)          


2017-03-30 | **[CV App]** Jun-Yan Zhu _et al._ **_"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"_**. **CycleGAN** [arXiv](https://arxiv.org/abs/1703.10593) [code](https://github.com/junyanz/CycleGAN) [project](https://junyanz.github.io/CycleGAN/) [PytorchCode](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)        
- Paper from `iGan` and `pix2pix` research group in UC Berkeley.        

2017-03-31 | **[Theory]** David Berthelot _et al._ **_"BEGAN: Boundary Equilibrium Generative Adversarial Networks"_**. **BEGAN** [arXiv](https://arxiv.org/abs/1703.10717) [code](https://github.com/wiseodd/generative-models)         


2017-03-31 | **[Music]** Li-Chia Yang _et al._ **"MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Condition"**. **MidiNet** [arXiv](https://arxiv.org/abs/1703.10847)


2017-03-31 | **[Theory]** Ishaan Gulrajani _et al._ **_"Improved Training of Wasserstein GANs"_** [arXiv](https://arxiv.org/abs/1704.00028) [code](https://github.com/wiseodd/generative-models)                
- This paper outlines the ways in which weight clipping in the discriminator can lead to pathological behavior which hurts stability and performance. Then, the paper proposes _WGAN with gradient penalty_, which does not suffer from the same issues, as an alternative.       
- The training of GAN is validated very stable and fast.  
- You **can** use `Adam` now! And `BatchNormalization` is no longer recommended in discriminator now based on the paper.        
***
2017-04-07 | **[CV App]** Weidong Yin _et al._ **"Semi-Latent GAN: Learning to generate and modify facial images from attributes"**. **Semi-Latent GAN** [arXiv](https://arxiv.org/abs/1704.02166)       


2017-04-08 | **[CV App]** Zili Yi _et al._ **"DualGAN: Unsupervised Dual Learning for Image-to-Image Translation"**. **MAGAN** [arXiv](https://arxiv.org/abs/1704.02510) [code](https://github.com/wiseodd/generative-models)         


2017-04-11 | **[CV App]** Xiaolong Wang _et al._ **"A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection"**. **A-Fast-RCNN** [arXiv(CVPR2017)](https://arxiv.org/abs/1704.03414) [code](https://github.com/xiaolonw/adversarial-frcnn)         


2017-04-12 | **[Theory]** Ruohan Wang _et al._ **"MAGAN: Margin Adaptation for Generative Adversarial Networks"**. **MAGAN** [arXiv](https://arxiv.org/abs/1704.03817) [code](https://github.com/wiseodd/generative-models)         
- This paper proposes a novel training procedure for Generative Adversarial Networks (GANs) to improve stability and performance by using an adaptive hinge loss objective function.        
- A simple and robust training procedure that adapts the hinge loss margin based on training statistics. The dependence on the margin hyper-parameter is removed and no new hyper-parameters to complicate training.        
- A principled analysis of the effects of the hinge loss margin on auto-encoder GANs training.        

2017-04-13 | **[CV App]** Rui Huang _et al._ **_"Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis"_**. [arXiv](https://arxiv.org/abs/1704.04086)        


2017-04-17 | **[CV App]** Bo Zhao _et al._ **"Multi-View Image Generation from a Single-View"**. [arXiv](https://arxiv.org/abs/1704.04886)


2017-04-17 | **[Theory]** Felix Juefei-Xu _et al._ **"Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking"**. **GoGAN** [arXiv](https://arxiv.org/abs/1704.04865)                    
- This work aims at improving on the `WGAN` by first generalizing its discriminator loss to a margin-based one, which leads to a better discriminator, and in turn a better generator, and then carrying out a progressive training paradigm involving multiple GANs to contribute to the maximum margin ranking loss so that the GAN at later stages will improve upon early stages.        
- `WGAN` loss treats a gap of 10 or 1 equally and it tries to increase the gap even further. The `MGAN` (Margin GAN, WGAN with margin-based discriminator loss proposed in the paper) loss will focus on increasing separation of examples with gap 1 and leave the samples with separation 10, which ensures a better discriminator, hence a better generator.        

2017-04-19 | **[CV App]** Yijun Li _et al._ **"Generative Face Completion"**. [arXiv(CVPR2017)](https://arxiv.org/abs/1704.05838) [code](https://github.com/Yijunmaverick/GenerativeFaceCompletion)         


2017-04-19 | **[CV App]** Jan Hendrik Metzen _et al._ **"Universal Adversarial Perturbations Against Semantic Image Segmentation"**. [arXiv](https://arxiv.org/abs/1704.05712)     


2017-04-20 | **[Theory]** Min Lin. **"Softmax GAN"**. **Softmax GAN** [arXiv](https://arxiv.org/abs/1704.06191)          
- Softmax GAN is a novel variant of Generative Adversarial Network (GAN). The key idea of Softmax GAN is to replace the classiﬁcation loss in the original GAN with a softmax cross-entropy loss in the sample space of one single batch.        

2017-04-24 | **[CV App]** Hengyue Pan and Hui Jiang. **"Supervised Adversarial Networks for Image Saliency Detection"**. [arXiv](https://arxiv.org/abs/1704.07242)          
***  
2017-05-02 | **[CV App]** Tseng-Hung Chen _et al._ **"Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner"**. [arXiv](https://arxiv.org/abs/1705.00930)       


2017-05-06 | **[CV App]** Zhimin Chen and Yuguang Tong. **"Face Super-Resolution Through Wasserstein GANs"**. [arXiv](https://arxiv.org/abs/1705.02438)       


2017-05-08 | **[CV App]** Jae Hyun Lim and Jong Chul Ye. **"Geometric GAN"**. [arXiv](https://arxiv.org/abs/1705.02894)       

2017-05-08 | **[CV App]** Qiangeng Xu _et al._ **"Generative Cooperative Net for Image Generation and Data Augmentation"**. [arXiv](https://arxiv.org/abs/1705.02887)       


2017-05-09 | **[Theory]** Hyeungill Lee _et al._ **"Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN"**. [arXiv](https://arxiv.org/abs/1705.03387)       


2017-05-14 | **[CV App]** Shuchang Zhou _et al._ **"GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data"**. **GeneGAN** [arXiv](https://arxiv.org/abs/1705.04932)      


2017-05-24 | **[Theory]** Aditya Grover _et al._ **"Flow-GAN: Bridging implicit and prescribed learning in generative models"**. **Flow-GAN** [arXiv](https://arxiv.org/abs/1705.08868)      

2017-05-24 | **[Theory]** Shuang Liu _et al._ **"Approximation and Convergence Properties of Generative Adversarial Learning"**. [arXiv](https://arxiv.org/abs/1705.08991)      


2017-05-25 | **[Theory]** Mathieu Sinn and Ambrish Rawat. **"Towards Consistency of Adversarial Training for Generative Models"**. [arXiv](https://arxiv.org/abs/1705.09199)      


2017-05-27 | **[Theory]** Zihang Dai _et al._ **"Good Semi-supervised Learning that Requires a Bad GAN"**. [arXiv](https://arxiv.org/abs/1705.09783)      


2017-05-31 | **[NLP]** Sai Rajeswar _et al._ **"Adversarial Generation of Natural Language"**. [arXiv](https://arxiv.org/abs/1705.10929)      
- This paper gave rise to a **"discussion"** between _Yoav Goldberg_ and _Yann LeCun_ about arXiv and researches of NLP. See [Yoav Goldberg's Medium](https://medium.com/@yoav.goldberg/an-adversarial-review-of-adversarial-generation-of-natural-language-409ac3378bd7), [Yann LeCun's Facebook](https://www.facebook.com/yann.lecun/posts/10154498539442143) and [Yoav Goldberg's response](https://medium.com/@yoav.goldberg/a-response-to-yann-lecuns-response-245125295c02) for details.
***
2017-06-02 | **[Theory]** Zhiting Hu _et al._ **"On Unifying Deep Generative Models"**. [arXiv](https://arxiv.org/abs/1706.00550)    

2017-06-05 | **[NLP]** Ofir Press _et al._ **"Language Generation with Recurrent Generative Adversarial Networks without Pre-training"**. [arXiv](https://arxiv.org/abs/1706.01399)     

2017-06-06 | **[Medical]** Yuan Xue _et al._ **"SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation"**. **SeqAN** [arXiv](https://arxiv.org/abs/1706.01805)     

2017-06-07 | **[Theory]** Swaminathan Gurumurthy _et al._ **"DeLiGAN:Generative Adversarial Networks for Diverse and Limited Data"**. **DeLiGAN** [arXiv](https://arxiv.org/abs/1706.02071) [code](https://github.com/val-iisc/deligan)    
