import React from 'react'
import Bad1 from './images/bad1.png'
import Bad2 from './images/bad2.png'
import Bad3 from './images/bad3.png'
import Good1 from './images/good1.png'
import Good2 from './images/good2.png'
import Good3 from './images/good3.png'

const About = () => (
  <div className="about-div">
    <h1 style={{ marginBottom: 35 }}>About</h1>
    <h2>TL;DR</h2>
    <p>We use a machine learning model called a GAN, that looks at a bunch of dog pictures, to determine the features of a dog. It then generates what it thinks is a dog, from a randomized seed.</p>
    <p>
      We teach another model to encode a dog image into a seed that generates an image similar to the original dog image. We call this process “gannifying”.
    </p>

    <h2 style={{ marginTop: 30 }}>THE GAN</h2>
    <h3>Basics</h3>
    <p>The primary technology behind this web application is a machine learning system known as a <b>Deep Convolutional Generative Adversarial Network (DCGAN).</b> This system trains two separate neural networks -- a generator and a discriminator -- with clashing objectives, hence, adversarial. The generator aims to create a normalized, fake image that is indistinguishable from a real image. Meanwhile, the discriminator aims to distinguish between real images, and fake images produced by the generator. The result is a system that slowly becomes better at both creating fake images, as well as spotting fake images. Check out <a target='_blank' rel='noopener noreferrer' href='https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29'>this article by Joseph Rocca</a> for further reading.</p>

    <h3>Preprocessing</h3>
    <p>Using the Stanford dogs dataset found <a href='http://vision.stanford.edu/aditya86/ImageNetDogs/' target='_blank' rel='noopener noreferrer'>here</a>, we trained a GAN system in PyTorch. As with any project, we began with preprocessing the dataset. First, we cropped every image, using a corresponding bounding box annotation, which was provided in the dataset. Then, before any image was fed into our discriminator, it was randomly cropped, resized to a 128x128 square, and finally normalized.</p>

    <h3>Training</h3>
    <p>In order to make our network more robust to noise, we added gradually decreasing noise values to our real images <i>(the noise reduced to zero by the 30th epoch).</i> In addition, as recommended by Jason Brownlee <a href='https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/' target='_blank' rel='noopener noreferrer'>here</a>, we used label smoothing while training the discriminator. In essence, we changed labels of 0 to a random value between 0 and 0.1, and labels of 1 to a random value between 0.9 and 1. This essentially handicaps the discriminator slightly, so it doesn’t improve so quickly as to overpower the generator.</p>
    <p>With these strategies that we employed, our generator produced images went from something like these:</p>
    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
      <img className='about-img' src={Bad1} alt='bad gen 1' />
      <img className='about-img' src={Bad2} alt='bad gen 2' />
      <img className='about-img' src={Bad3} alt='bad gen 3' />
    </div>
    <p>To these!</p>
    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
      <img className='about-img' src={Good1} alt='good gen 1' />
      <img className='about-img' src={Good2} alt='good gen 2' />
      <img className='about-img' src={Good3} alt='good gen 3' />
    </div>

    <h2>THE ENCODER</h2>
    <h3>Basics</h3>
    <p>An encoder is a third network that looks like the discriminator. It takes a real image as input, and outputs a seed that can then be input into the generator, to produce a fake dog image that should resemble the original real image.</p>
    
    <h3>Training</h3>
    <p>To train the encoder, the output seed generated from the input real image is fed through the pretrained generator. Then, the generated fake image is compared to the original real image. This results in an autoencoder-esque network that learns how to generate a fake image with similar features <i>(structural, color, etc.)</i> to the input image. Because this produces an image stylistically similar to those produced by GANs, we call it <em>“gannifying”.</em></p>

    <h2>THE APP</h2>
    <p>Finally, we packaged all of our models into a Flask server, and created a React frontend for an interactive experience. Then, everything was wrapped with Docker and deployed on AWS using ElasticBeanstalk and Cloudfront.</p>

    <h2>FURTHER IDEAS</h2>
    <p>As of now, our generator and encoder does seem to create images that suggest the idea of dogs, including textures, colors, and general structures, but often fails to mix these aspects in a cohesive manner. Some ideas that could be explored further in order to improve the performance of the generator include:</p>
    <ul>
      <li>Using a more modern network</li>
      <li>Employing more improvement strategies from Jason Brownlee</li>
      <li>Training for more epochs</li>
      <li>Looking into image augmentations <i>(color jitter, style augmentation, etc.)</i></li>
    </ul>

  </div>
)

export default About
