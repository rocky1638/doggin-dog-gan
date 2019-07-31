from torch import cuda
import torch
import tqdm
from dataset import GenerateIterator
import matplotlib.pyplot as plt
from models import Generator, Discriminator
from generate import generate_image
from options import opts
from models import weights_init

if __name__ == "__main__":

	# Loss function
	adversarial_loss = torch.nn.BCELoss()

	# init models and apply custom weight initialization
	gen = Generator()
	gen.apply(weights_init)

	disc = Discriminator()
	disc.apply(weights_init)

	# train from previously created model
	if opts.continueTrain:
		gen.load_state_dict(torch.load('./generator.pt'))
		disc.load_state_dict(torch.load('./discriminator.pt'))

	optimizer_g = torch.optim.Adam(gen.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
	optimizer_d = torch.optim.Adam(disc.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

	image_iterator = GenerateIterator('./images_cropped', shuffle=True)

	# fix the noise so that we can see the image generated on the same noise every epoch
	fixed_noise = torch.randn(opts.outputNum, opts.noiseSize, 1, 1)

	if torch.cuda.is_available():
		adversarial_loss = adversarial_loss.cuda()
		gen = gen.cuda()
		disc = disc.cuda()

	start_epoch = 1
	losses = [[], []]

	for epoch in range(start_epoch, opts.numEpochs):
		progress_bar = tqdm.tqdm(image_iterator)
		total_gen_loss = 0
		total_disc_loss = 0

		# slowly anneal the standard deviation of the noise added to input, until it is removed fully at 30 epochs
		standard_dev = 0.1 - (0.1/30) * epoch
		if standard_dev < 0:
			standard_dev = 0
		image_iterator.dataset.std = standard_dev

		for batch_num, images in enumerate(progress_bar):
			fake_image_labels = torch.ones(images.shape[0],)
			valid_image_labels = torch.zeros(images.shape[0],)

			# try disc with soft labels, fake is between 0.9, 1.0, real between 0.0, 0.1
			fake_image_labels_d = torch.rand(images.shape[0]) / 10 + 0.9
			valid_image_labels_d = torch.rand(images.shape[0]) / 10

			seed = torch.randn(images.shape[0], opts.noiseSize, 1, 1)

			if torch.cuda.is_available():
				valid_image_labels = valid_image_labels.cuda()
				fake_image_labels = fake_image_labels.cuda()
				valid_image_labels_d = valid_image_labels_d.cuda()
				fake_image_labels_d = fake_image_labels_d.cuda()
				images = images.cuda()
				seed = seed.cuda()

			# ----- train generator -----
			fake_images = gen(seed)
			gen_loss = adversarial_loss(disc(fake_images), valid_image_labels)
			total_gen_loss += gen_loss.data

			optimizer_g.zero_grad()
			gen_loss.backward()
			optimizer_g.step()

			# ----- train discriminator -----
			pred_real = disc(images)
			pred_fake = disc(fake_images.detach())
			real_loss = adversarial_loss(pred_real, valid_image_labels_d)
			fake_loss = adversarial_loss(pred_fake, fake_image_labels_d)

			# divide by 2 or else loss is weighted 2x towards disc vs gen
			disc_loss = (real_loss + fake_loss) / 2
			total_disc_loss += disc_loss.data

			optimizer_d.zero_grad()
			disc_loss.backward()
			optimizer_d.step()

			progress_bar.set_description(
				f"epoch: {epoch} || disc loss: {total_disc_loss/batch_num} || gen loss: {total_gen_loss/batch_num}"
			)

		denom = opts.numImages // opts.batchSize

		avg_gen_loss = total_gen_loss / denom
		avg_disc_loss = total_disc_loss / denom
		losses[0].append(avg_gen_loss)
		losses[1].append(avg_disc_loss)

		torch.save(gen.state_dict(), './generator.pt')
		torch.save(disc.state_dict(), './discriminator.pt')
		generate_image(fixed_noise, epoch=epoch)

	# display loss graph
	plt.plot(losses[0], label="gen")
	plt.plot(losses[1], label="disc")
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(loc='upper left')
	plt.show()

