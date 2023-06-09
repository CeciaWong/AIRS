import signal
import argparse
import itertools
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import *
import torch
import datetime
import platform
import os 

if __name__ == '__main__':
    #signal.signal(signal.SIGINT, emergency_stop)
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    device_ids = range(torch.cuda.device_count())
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="3cGAN")
    parser.add_argument("--ismerged", type=int, default=1)
    parser.add_argument("--mergew", type=str, default="100000")
    parser.add_argument("--save_name", type=str, default=datetime.datetime.now().strftime("%m%d%H%M"), help="name of the network")
    parser.add_argument("--network_name", type=str, default="3cGAN", help="name of the network")
    parser.add_argument("--training_dataset", type=str, default="osteo", help="name of the dataset")
    parser.add_argument("--testing_dataset", type=str, default="osteo", help="name of the testing dataset")
    parser.add_argument("--lambda_merging", type=float, default=10, help="scaling factor for the new loss")
    parser.add_argument("--lambda_cyc", type=float, default=1, help="cycle loss weight")

    parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs oef training")
    parser.add_argument("--batch_size", type=int, default=1*len(device_ids), help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=200, help="size of image height")
    parser.add_argument("--img_width", type=int, default=200, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
    parser.add_argument("--textfile_training_results_interval", type=int, default=50, help="textfile_training_results_interval")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    opt = parser.parse_args()

    print(opt)
    print("Begin time:"+datetime.datetime.now().strftime("%y-%m-%d %H:%M"))
    # Create sample and checkpoint directories
    os.makedirs("../3cGAN_saved_models/%s-%s-%s" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset), exist_ok=True)
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_CB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BC = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_AC = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_CA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_B1 = Discriminator(input_shape)
    D_A2 = Discriminator(input_shape)
    D_B3 = Discriminator(input_shape)
    D_C4 = Discriminator(input_shape)
    D_C5 = Discriminator(input_shape)
    D_A6 = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        G_CB = G_CB.cuda()
        G_BC = G_BC.cuda()
        G_AC = G_AC.cuda()
        G_CA = G_CA.cuda()

        D_B1 = D_B1.cuda()
        D_A2 = D_A2.cuda()
        D_B3 = D_B3.cuda()
        D_C4 = D_C4.cuda()
        D_C5 = D_C5.cuda()
        D_A6 = D_A6.cuda()

        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    # using multiple GPUs
    if len(device_ids)>1:
        G_AB = nn.DataParallel(G_AB,device_ids)
        G_BA = nn.DataParallel(G_BA,device_ids)
        G_CB = nn.DataParallel(G_CB,device_ids)
        G_BC = nn.DataParallel(G_BC,device_ids)
        G_AC = nn.DataParallel(G_AC,device_ids)
        G_CA = nn.DataParallel(G_CA,device_ids)

        D_B1 = nn.DataParallel(D_B1,device_ids)
        D_A2 = nn.DataParallel(D_A2,device_ids)
        D_B3 = nn.DataParallel(D_B3,device_ids)
        D_C4 = nn.DataParallel(D_C4,device_ids)
        D_C5 = nn.DataParallel(D_C5,device_ids)
        D_A6 = nn.DataParallel(D_A6,device_ids)

    # if opt.epoch != 1:
    #     # Load pretrained models
    #     G_AB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AB_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     G_BA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BA_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     G_CB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CB_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     G_BC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BC_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     G_AC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AC_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     G_CA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CA_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))

    #     D_B1.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_B1_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     D_A2.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_A2_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     D_B3.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_B3_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     D_C4.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_C4_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     D_C5.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_C5_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    #     D_A6.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_A6_%d.pth" % (opt.save_name, opt.network_name, opt.training_dataset, opt.epoch)))
    # else:
    #     # Initialize weights
    #     G_AB.apply(weights_init_normal)
    #     G_BA.apply(weights_init_normal)
    #     G_CB.apply(weights_init_normal)
    #     G_BC.apply(weights_init_normal)
    #     G_AC.apply(weights_init_normal)
    #     G_CA.apply(weights_init_normal)

    #     D_B1.apply(weights_init_normal)
    #     D_A2.apply(weights_init_normal)
    #     D_B3.apply(weights_init_normal)
    #     D_C4.apply(weights_init_normal)
    #     D_C5.apply(weights_init_normal)
    #     D_A6.apply(weights_init_normal)

    # Optimizers
    """
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters(), G_CB.parameters(), G_BC.parameters(), G_AC.parameters(), G_CA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_B1 = torch.optim.Adam(D_B1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A2 = torch.optim.Adam(D_A2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B3 = torch.optim.Adam(D_B3.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_C4 = torch.optim.Adam(D_C4.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_C5 = torch.optim.Adam(D_C5.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A6 = torch.optim.Adam(D_A6.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A2 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B3 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B3, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_C4 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_C4, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_C5 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_C5, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A6 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A6, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_B1_buffer = ReplayBuffer()
    fake_A2_buffer = ReplayBuffer()
    fake_B3_buffer = ReplayBuffer()
    fake_C4_buffer = ReplayBuffer()
    fake_C5_buffer = ReplayBuffer()
    fake_A6_buffer = ReplayBuffer()


    # Image transformations
    transforms_ = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]


    dataloader = DataLoader(
        ImageDataset("Training/%s-training" % opt.training_dataset, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    transforms_testing_non_fliped_ = [
        transforms.ToTensor(),
    ]

    val_dataloader_non_flipped = DataLoader(
        ImageDataset("Testing/%s-testing" % opt.training_dataset, transforms_=transforms_testing_non_fliped_, unaligned=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    
    for epoch in range(opt.epoch, opt.n_epochs+1):
        G_AB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AB_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        G_BA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BA_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        G_CB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CB_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        G_BC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BC_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        G_AC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AC_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        G_CA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CA_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))

        D_B1.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_B1_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        D_A2.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_A2_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        D_B3.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_B3_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        D_C4.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_C4_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        D_C5.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_C5_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
        D_A6.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/D_A6_%dep.pth" % (opt.save_name.strip("_val"), opt.network_name, opt.training_dataset, epoch)))
    
        """
        for i, batch in enumerate(dataloader):
            # Set model input
            if len(device_ids)<=1:
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))
                real_C = Variable(batch["C"].type(Tensor))
            else:
                real_A = Variable(batch["A"].cuda().type(Tensor))
                real_B = Variable(batch["B"].cuda().type(Tensor))
                real_C = Variable(batch["C"].cuda().type(Tensor))
            
            # Adversarial ground truths
            if len(device_ids) <= 1:
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
            else:
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.module.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A2.module.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()
            G_CB.train()
            G_BC.train()
            G_AC.train()
            G_CA.train()

            optimizer_G.zero_grad()

            # GAN loss
            fake_B1 = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B1(fake_B1), valid)
            fake_A2 = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A2(fake_A2), valid)

            fake_B3 = G_CB(real_C)
            loss_GAN_CB = criterion_GAN(D_B3(fake_B3), valid)
            fake_C4 = G_BC(real_B)
            loss_GAN_BC = criterion_GAN(D_C4(fake_C4), valid)

            fake_C5 = G_AC(real_A)
            loss_GAN_AC = criterion_GAN(D_C5(fake_C5), valid)
            fake_A6 = G_CA(real_C)
            loss_GAN_CA = criterion_GAN(D_A6(fake_A6), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA + loss_GAN_BC + loss_GAN_CB + loss_GAN_AC + loss_GAN_BC) / 6

            # Cycle loss
            recov_BA = G_BA(fake_B1)
            loss_cycle_BA = criterion_cycle(recov_BA, real_A)
            recov_AB = G_AB(fake_A2)
            loss_cycle_AB = criterion_cycle(recov_AB, real_B)

            recov_BC = G_BC(fake_B3)
            loss_cycle_BC = criterion_cycle(recov_BC, real_C)
            recov_CB = G_CB(fake_C4)
            loss_cycle_CB = criterion_cycle(recov_CB, real_B)

            recov_AC = G_AC(fake_A6)
            loss_cycle_AC = criterion_cycle(recov_AC, real_C)
            recov_CA = G_CA(fake_C5)
            loss_cycle_CA = criterion_cycle(recov_CA, real_A)


            # merging loss:
            recov_253461 = G_AB(G_CA(G_BC(G_CB(G_AC(G_BA(real_B))))))
            loss_cycle_253461 = criterion_cycle(recov_253461, real_B)


            loss_cycle = (loss_cycle_BA + loss_cycle_AB + loss_cycle_BC + loss_cycle_CB + loss_cycle_CA + loss_cycle_AC) / 6 + opt.lambda_merging*loss_cycle_253461

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A2
            # -----------------------

            optimizer_D_A2.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A2(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A2_ = fake_A2_buffer.push_and_pop(fake_A2)
            loss_fake = criterion_GAN(D_A2(fake_A2_.detach()), fake)
            # Total loss
            loss_D_A2 = (loss_real + loss_fake) / 2

            loss_D_A2.backward()
            optimizer_D_A2.step()

            # -----------------------
            #  Train Discriminator B1
            # -----------------------

            optimizer_D_B1.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B1(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B1_ = fake_B1_buffer.push_and_pop(fake_B1)
            loss_fake = criterion_GAN(D_B1(fake_B1_.detach()), fake)
            # Total loss
            loss_D_B1 = (loss_real + loss_fake) / 2

            loss_D_B1.backward()
            optimizer_D_B1.step()

            # -----------------------
            #  Train Discriminator B3
            # -----------------------

            optimizer_D_B3.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B3(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B3_ = fake_B3_buffer.push_and_pop(fake_B3)
            loss_fake = criterion_GAN(D_B3(fake_B3_.detach()), fake)
            # Total loss
            loss_D_B3 = (loss_real + loss_fake) / 2

            loss_D_B3.backward()
            optimizer_D_B3.step()

            # -----------------------
            #  Train Discriminator C4
            # -----------------------

            optimizer_D_C4.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C4(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C4_ = fake_C4_buffer.push_and_pop(fake_C4)
            loss_fake = criterion_GAN(D_C4(fake_C4_.detach()), fake)
            # Total loss
            loss_D_C4 = (loss_real + loss_fake) / 2

            loss_D_C4.backward()
            optimizer_D_C4.step()


            # -----------------------
            #  Train Discriminator C5
            # -----------------------

            optimizer_D_C5.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C5(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C5_ = fake_C5_buffer.push_and_pop(fake_C5)
            loss_fake = criterion_GAN(D_C5(fake_C5_.detach()), fake)
            # Total loss
            loss_D_C5 = (loss_real + loss_fake) / 2

            loss_D_C5.backward()
            optimizer_D_C5.step()

            # -----------------------
            #  Train Discriminator A6
            # -----------------------

            optimizer_D_A6.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A6(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A6_ = fake_A6_buffer.push_and_pop(fake_A6)
            loss_fake = criterion_GAN(D_A6(fake_A6_.detach()), fake)
            # Total loss
            loss_D_A6 = (loss_real + loss_fake) / 2

            loss_D_A6.backward()
            optimizer_D_A6.step()

            loss_D = (loss_D_A2 + loss_D_B1 + loss_D_B3 + loss_D_C4 + loss_D_C5 + loss_D_A6) / 6

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    opt.lambda_cyc*loss_cycle.item(),
                    #loss_identity.item(),
                    time_left,
                )
            )

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A2.step()
        lr_scheduler_D_B1.step()
        lr_scheduler_D_B3.step()
        lr_scheduler_D_C4.step()
        lr_scheduler_D_C5.step()
        lr_scheduler_D_A6.step()
        """
        # validation below
        for i, batch in enumerate(val_dataloader_non_flipped):
            # Set model input
            if len(device_ids)<=1:
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))
                real_C = Variable(batch["C"].type(Tensor))
            else:
                real_A = Variable(batch["A"].cuda().type(Tensor))
                real_B = Variable(batch["B"].cuda().type(Tensor))
                real_C = Variable(batch["C"].cuda().type(Tensor))
            
            # Adversarial ground truths
            if len(device_ids) <= 1:
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
            else:
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.module.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A2.module.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.eval()
            G_BA.eval()
            G_CB.eval()
            G_BC.eval()
            G_AC.eval()
            G_CA.eval()

            # GAN loss
            fake_B1 = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B1(fake_B1), valid)
            fake_A2 = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A2(fake_A2), valid)

            fake_B3 = G_CB(real_C)
            loss_GAN_CB = criterion_GAN(D_B3(fake_B3), valid)
            fake_C4 = G_BC(real_B)
            loss_GAN_BC = criterion_GAN(D_C4(fake_C4), valid)

            fake_C5 = G_AC(real_A)
            loss_GAN_AC = criterion_GAN(D_C5(fake_C5), valid)
            fake_A6 = G_CA(real_C)
            loss_GAN_CA = criterion_GAN(D_A6(fake_A6), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA + loss_GAN_BC + loss_GAN_CB + loss_GAN_AC + loss_GAN_BC) / 6

            # Cycle loss
            recov_BA = G_BA(fake_B1)
            loss_cycle_BA = criterion_cycle(recov_BA, real_A)
            recov_AB = G_AB(fake_A2)
            loss_cycle_AB = criterion_cycle(recov_AB, real_B)

            recov_BC = G_BC(fake_B3)
            loss_cycle_BC = criterion_cycle(recov_BC, real_C)
            recov_CB = G_CB(fake_C4)
            loss_cycle_CB = criterion_cycle(recov_CB, real_B)

            recov_AC = G_AC(fake_A6)
            loss_cycle_AC = criterion_cycle(recov_AC, real_C)
            recov_CA = G_CA(fake_C5)
            loss_cycle_CA = criterion_cycle(recov_CA, real_A)

            # merging loss:
            loss_cycle0 = (loss_cycle_BA + loss_cycle_AB + loss_cycle_BC + loss_cycle_CB + loss_cycle_CA + loss_cycle_AC) / 6 
            if opt.ismerged:
                # merging loss:
                loss_cycle_mergedlist = []
                if int(list(opt.mergew)[0]):
                    recov_mAB = G_AB(G_CA(G_BC(G_CB(G_AC(G_BA(real_B))))))
                    loss_cycle_mAB = criterion_cycle(recov_mAB, real_B)
                    loss_cycle_mergedlist.append(loss_cycle_mAB)
                if int(list(opt.mergew)[1]):
                    recov_mAC = G_AC(G_BA(G_CB(G_BC(G_AB(G_CA(real_C))))))
                    loss_cycle_mAC = criterion_cycle(recov_mAC, real_C)
                    loss_cycle_mergedlist.append(loss_cycle_mAC)
                if int(list(opt.mergew)[2]):
                    recov_mBA = G_BA(G_CB(G_AC(G_CA(G_BC(G_AB(real_A))))))
                    loss_cycle_mBA = criterion_cycle(recov_mBA, real_A)
                    loss_cycle_mergedlist.append(loss_cycle_mBA)
                if int(list(opt.mergew)[3]):
                    recov_mBC = G_BC(G_AB(G_CA(G_AC(G_BA(G_CB(real_C))))))
                    loss_cycle_mBC = criterion_cycle(recov_mBC, real_C)
                    loss_cycle_mergedlist.append(loss_cycle_mBC)
                if int(list(opt.mergew)[4]):
                    recov_mCA = G_CA(G_BC(G_AB(G_BA(G_CB(G_AC(real_A))))))
                    loss_cycle_mCA = criterion_cycle(recov_mCA, real_A)
                    loss_cycle_mergedlist.append(loss_cycle_mCA)
                if int(list(opt.mergew)[5]):
                    recov_mCB = G_CB(G_AC(G_BA(G_AB(G_CA(G_BC(real_B))))))
                    loss_cycle_mCB = criterion_cycle(recov_mCB, real_B)
                    loss_cycle_mergedlist.append(loss_cycle_mCB)
                loss_cycle_merged = sum(loss_cycle_mergedlist)/len(loss_cycle_mergedlist)
                loss_cycle = loss_cycle0 + opt.lambda_merging*loss_cycle_merged
            else:
                loss_cycle = loss_cycle0
            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle
            loss_G.backward()
            #optimizer_G.step()

            # -----------------------
            #  Train Discriminator A2
            # -----------------------

            #optimizer_D_A2.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A2(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A2_ = fake_A2_buffer.push_and_pop(fake_A2)
            loss_fake = criterion_GAN(D_A2(fake_A2_.detach()), fake)
            # Total loss
            loss_D_A2 = (loss_real + loss_fake) / 2

            loss_D_A2.backward()
            #optimizer_D_A2.step()

            # -----------------------
            #  Train Discriminator B1
            # -----------------------

            #optimizer_D_B1.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B1(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B1_ = fake_B1_buffer.push_and_pop(fake_B1)
            loss_fake = criterion_GAN(D_B1(fake_B1_.detach()), fake)
            # Total loss
            loss_D_B1 = (loss_real + loss_fake) / 2

            loss_D_B1.backward()
            #optimizer_D_B1.step()

            # -----------------------
            #  Train Discriminator B3
            # -----------------------

            #optimizer_D_B3.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B3(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B3_ = fake_B3_buffer.push_and_pop(fake_B3)
            loss_fake = criterion_GAN(D_B3(fake_B3_.detach()), fake)
            # Total loss
            loss_D_B3 = (loss_real + loss_fake) / 2

            loss_D_B3.backward()
            #optimizer_D_B3.step()

            # -----------------------
            #  Train Discriminator C4
            # -----------------------

            #optimizer_D_C4.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C4(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C4_ = fake_C4_buffer.push_and_pop(fake_C4)
            loss_fake = criterion_GAN(D_C4(fake_C4_.detach()), fake)
            # Total loss
            loss_D_C4 = (loss_real + loss_fake) / 2

            loss_D_C4.backward()
            #optimizer_D_C4.step()


            # -----------------------
            #  Train Discriminator C5
            # -----------------------

            #optimizer_D_C5.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C5(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C5_ = fake_C5_buffer.push_and_pop(fake_C5)
            loss_fake = criterion_GAN(D_C5(fake_C5_.detach()), fake)
            # Total loss
            loss_D_C5 = (loss_real + loss_fake) / 2

            loss_D_C5.backward()
            #optimizer_D_C5.step()

            # -----------------------
            #  Train Discriminator A6
            # -----------------------

            #optimizer_D_A6.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A6(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A6_ = fake_A6_buffer.push_and_pop(fake_A6)
            loss_fake = criterion_GAN(D_A6(fake_A6_.detach()), fake)
            # Total loss
            loss_D_A6 = (loss_real + loss_fake) / 2

            loss_D_A6.backward()
            #optimizer_D_A6.step()

            loss_D = (loss_D_A2 + loss_D_B1 + loss_D_B3 + loss_D_C4 + loss_D_C5 + loss_D_A6) / 6

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if opt.ismerged:
                sys.stdout.write(
                    "\r*[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, cycle0: %f, mcycle: %f] \
                        [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        opt.lambda_cyc*loss_cycle.item(),
                        opt.lambda_cyc*loss_cycle0.item(),
                        opt.lambda_cyc*opt.lambda_merging*loss_cycle_merged.item(),

                        loss_D_B1.item(),
                        opt.lambda_cyc*loss_cycle_AB.item()+loss_GAN_AB.item(),
                        loss_GAN_AB.item(),
                        opt.lambda_cyc*loss_cycle_AB.item(),

                        loss_D_A2.item(),
                        opt.lambda_cyc*loss_cycle_BA.item()+loss_GAN_BA.item(),
                        loss_GAN_BA.item(),
                        opt.lambda_cyc*loss_cycle_BA.item(),

                        loss_D_B3.item(),
                        opt.lambda_cyc*loss_cycle_CB.item()+loss_GAN_CB.item(),
                        loss_GAN_CB.item(),
                        opt.lambda_cyc*loss_cycle_CB.item(),

                        loss_D_C4.item(),
                        opt.lambda_cyc*loss_cycle_BC.item()+loss_GAN_BC.item(),
                        loss_GAN_BC.item(),
                        opt.lambda_cyc*loss_cycle_BC.item(),

                        loss_D_C5.item(),
                        opt.lambda_cyc*loss_cycle_AC.item()+loss_GAN_AC.item(),
                        loss_GAN_AC.item(),
                        opt.lambda_cyc*loss_cycle_AC.item(),
                        
                        loss_D_A6.item(),
                        opt.lambda_cyc*loss_cycle_CA.item()+loss_GAN_CA.item(),
                        loss_GAN_CA.item(),
                        opt.lambda_cyc*loss_cycle_CA.item(),

                        time_left
                    )
                )
            else:
                sys.stdout.write(
                    "\r*[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, cycle0: %f, mcycle: %f] \
                        [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] [%f, %f, %f, %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        0, #opt.lambda_cyc*loss_cycle.item(),
                        opt.lambda_cyc*loss_cycle0.item(),
                        0,

                        loss_D_B1.item(),
                        opt.lambda_cyc*loss_cycle_AB.item()+loss_GAN_AB.item(),
                        loss_GAN_AB.item(),
                        opt.lambda_cyc*loss_cycle_AB.item(),

                        loss_D_A2.item(),
                        opt.lambda_cyc*loss_cycle_BA.item()+loss_GAN_BA.item(),
                        loss_GAN_BA.item(),
                        opt.lambda_cyc*loss_cycle_BA.item(),

                        loss_D_B3.item(),
                        opt.lambda_cyc*loss_cycle_CB.item()+loss_GAN_CB.item(),
                        loss_GAN_CB.item(),
                        opt.lambda_cyc*loss_cycle_CB.item(),

                        loss_D_C4.item(),
                        opt.lambda_cyc*loss_cycle_BC.item()+loss_GAN_BC.item(),
                        loss_GAN_BC.item(),
                        opt.lambda_cyc*loss_cycle_BC.item(),

                        loss_D_C5.item(),
                        opt.lambda_cyc*loss_cycle_AC.item()+loss_GAN_AC.item(),
                        loss_GAN_AC.item(),
                        opt.lambda_cyc*loss_cycle_AC.item(),
                        
                        loss_D_A6.item(),
                        opt.lambda_cyc*loss_cycle_CA.item()+loss_GAN_CA.item(),
                        loss_GAN_CA.item(),
                        opt.lambda_cyc*loss_cycle_CA.item(),
                        
                        time_left
                    )
                )
        
        # if not opt.checkpoint_interval != -1 and (epoch % opt.checkpoint_interval == 0 or opt.n_epochs - epoch < 5):
        #     continue
        
    print("End time:"+datetime.datetime.now().strftime("%y-%m-%d %H:%M"))
