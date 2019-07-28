if __name__ == "__main__":

	import numpy as np
	from torch import optim, nn, cuda, LongTensor, FloatTensor, ones, zeros, from_numpy, save
	import tqdm
	from dataset import GenerateIterator
	import matplotlib.pyplot as plt
	from models import Generator, Discriminator
	from generate import generate_image
	from options import opts
	

	# Loss function
	adversarial_loss = nn.CrossEntropyLoss()

	gen = Generator()
	disc = Discriminator()

	optimizer_g = optim.Adam(gen.parameters(), lr=0.01, betas=(0.9,0.999))
	optimizer_d = optim.Adam(disc.parameters(), lr=0.01, betas=(0.9,0.999))

	image_iterator = GenerateIterator('./images_cropped')

	if cuda.is_available():
	    adversarial_loss = adversarial_loss.cuda()
	    gen = gen.cuda()
	    disc = disc.cuda()
	    
	start_epoch = 1
	losses = [[],[]]

	for epoch in range(start_epoch,opts.numEpochs):
	    progress_bar = tqdm.tqdm(image_iterator)
	    total_gen_loss = 0
	    total_disc_loss = 0

	    for images in progress_bar:
	        valid_image_labels = ones(images.shape[0]).type(LongTensor)
	        fake_image_labels = zeros(images.shape[0]).type(LongTensor)

	        seed = from_numpy(np.random.normal(0,1, size=(images.shape[0], 100))).type(FloatTensor)
	        if cuda.is_available():
	            valid_image_labels, fake_image_labels, images, seed = valid_image_labels.cuda(), fake_image_labels.cuda(), images.cuda(), seed.cuda()
	        
	        # --- train discriminator ---
	        optimizer_d.zero_grad()

	        real_loss = adversarial_loss(disc(images), valid_image_labels)
	       	real_loss.backward()

	        fake_images = gen(seed).detach()
	        fake_loss = adversarial_loss(disc(fake_images), fake_image_labels)
	        fake_loss.backward()

	        # divide by 2 or else loss is weighted 2x towards disc vs gen
	        disc_loss = (real_loss + fake_loss) / 2
	        total_disc_loss += disc_loss.data
	        
	        optimizer_d.step()

	       	# --- train generator ---
	       	disc = disc.eval()
	       	
	       	optimizer_g.zero_grad()
	       	
	        gen_loss = adversarial_loss(disc(fake_images), valid_image_labels)
	        total_gen_loss += gen_loss.data
	        
	        gen_loss.backward()
	        optimizer_g.step()

	        disc = disc.train()
	        
	        progress_bar.set_description(f"disc loss: {disc_loss} || gen loss: {gen_loss}")

	    denom = opts.numImages // opts.batchSize

	    avg_gen_loss = total_gen_loss / denom
	    avg_disc_loss = total_disc_loss / denom
	    losses[0].append(avg_gen_loss)
	    losses[1].append(avg_disc_loss)

	    save(gen.state_dict(), './generator.pt')
	    generate_image('./generator.pt', epoch=epoch)


	# display loss graph
	plt.plot(losses[0], label="gen")
	plt.plot(losses[1], label="disc")
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(loc='upper left')
	plt.show()

	save(gen.state_dict(), './generator.pt')