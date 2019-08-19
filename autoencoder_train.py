from torch import cuda
import torch
import tqdm
from dataset import GenerateIterator
from models import Generator, Imageencoder
from options import opts
from models import weights_init
from torchvision.utils import save_image

if __name__ == "__main__":

	# Loss function
	adversarial_loss = torch.nn.MSELoss()

	# init models and apply custom weight initialization
	gen = Generator()
	gen.apply(weights_init)

	encoder = Imageencoder()
	encoder.apply(weights_init)

	# train from previously created model
	if opts.continueTrain:
		encoder.load_state_dict(torch.load('./encoder.pt'))
	gen.load_state_dict(torch.load('./generator.pt'))

	optimizer = torch.optim.Adam(encoder.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
	image_iterator = GenerateIterator('./images_cropped', shuffle=True)

	if cuda.is_available():
		adversarial_loss = adversarial_loss.cuda()
		gen = gen.cuda()
		encoder = encoder.cuda()

	start_epoch = 1
	losses = [[], []]

	# gen = gen.eval()

	for epoch in range(start_epoch, opts.numEpochs):
		progress_bar = tqdm.tqdm(image_iterator)
		total_gen_loss = 0
		total_disc_loss = 0

		# slowly anneal the standard deviation of the noise added to input, until it is removed fully at 30 epochs
		standard_dev = 0
		image_iterator.dataset.std = standard_dev

		for batch_num, images in enumerate(progress_bar):

			if cuda.is_available():
				images = images.cuda()

			# ----- train encoder -----
			encoded_noise = encoder(images)
			gen_image = gen(encoded_noise)
			imago = images
			loss = adversarial_loss(gen_image, images)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			progress_bar.set_description(
				f"epoch: {epoch} || loss: {loss.data} "
			)
		# quick image test
		save_image(gen_image.data[0], "./test0" + ".png", normalize=True)
		save_image(gen_image.data[1], "./test2" + ".png", normalize=True)
		save_image(gen_image.data[2], "./test4" + ".png", normalize=True)

		save_image(imago.cpu().data[0], "./test1" + ".png", normalize=True)
		save_image(imago.cpu().data[1], "./test3" + ".png", normalize=True)
		save_image(imago.cpu().data[2], "./test5" + ".png", normalize=True)

		torch.save(encoder.state_dict(), './encoder.pt')


