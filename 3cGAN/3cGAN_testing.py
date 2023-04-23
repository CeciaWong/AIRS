import argparse
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import *
import torch
from datetime import datetime
import numpy as np
print(torch.__version__)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_ids = range(torch.cuda.device_count())

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()

""" file folder's name of models to be tested """
parser.add_argument("--test_model", type=str, default="3cGANunalighned")#datetime.now().strftime("%m%d%H%M"), help="name of the network")
""" test epoch """
parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")

parser.add_argument("--network_name", type=str, default="3cGAN", help="name of the network")
parser.add_argument("--dataset_name", type=str, default="osteo", help="name of the training dataset")
parser.add_argument("--testing_dataset", type=str, default="osteo", help="name of the testing dataset")
parser.add_argument("--lambda_cyc", type=float, default=0.1, help="cycle loss weight")

parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
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
parser.add_argument("--lambda_id", type=float, default=1, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss(size_average=None, reduce=None, reduction = 'elementwise_mean')

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_AC = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_CA = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_CB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BC = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    G_AC = G_AC.cuda()
    G_CA = G_CA.cuda()
    G_CB = G_CB.cuda()
    G_BC = G_BC.cuda()
if len(device_ids)>1:
    G_AB = nn.DataParallel(G_AB,device_ids)
    G_BA = nn.DataParallel(G_BA,device_ids)
    G_AC = nn.DataParallel(G_AC,device_ids)
    G_CA = nn.DataParallel(G_CA,device_ids)
    G_CB = nn.DataParallel(G_CB,device_ids)
    G_BC = nn.DataParallel(G_BC,device_ids)

G_AB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AB_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))
G_BA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BA_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))
G_AC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_AC_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))
G_CA.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CA_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))
G_CB.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_CB_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))
G_BC.load_state_dict(torch.load("../3cGAN_saved_models/%s-%s-%s/G_BC_%dep.pth" % (opt.test_model, opt.network_name, opt.dataset_name, opt.epoch)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
# fake_B1_buffer = ReplayBuffer()
# fake_A2_buffer = ReplayBuffer()

transforms_testing_non_fliped_ = [
    transforms.ToTensor(),
]

val_dataloader_non_flipped = DataLoader(
    ImageDataset("Testing/%s-testing" % opt.testing_dataset, transforms_=transforms_testing_non_fliped_, unaligned=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


def testing():
    #os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-fakeB" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    #os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-fakeBtoA" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-display" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-AtoB" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-AtoC" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-BtoA" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-BtoC" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-CtoA" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    os.makedirs("../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-CtoB" % (opt.test_model, opt.epoch, opt.network_name), exist_ok=True)
    
    G_AB.eval()
    G_BA.eval()
    G_AC.eval()
    G_CA.eval()
    G_CB.eval()
    G_BC.eval()

    for i, batch in enumerate(val_dataloader_non_flipped):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        real_C = Variable(batch['C'].type(Tensor))
        num = str(batch['num']).strip("['").strip("']")
        "A to BC"
        fake_B1 = G_AB(real_A)
        fake_C5 = G_AC(real_A)
        fake_B1toA = G_BA(fake_B1)
        fake_C5toA = G_CA(fake_C5)
        "B to AC"
        fake_A2 = G_BA(real_B)
        fake_C4 = G_BC(real_B)
        fake_A2toB = G_AB(fake_A2)
        fake_C4toB = G_CB(fake_C4)
        "C to AB"
        fake_A6 = G_CA(real_C)
        fake_B3 = G_CB(real_C)
        fake_A6toC = G_AC(fake_A6)
        fake_B3toC = G_BC(fake_B3)



        display_imgA = torch.stack([real_A.squeeze(0).cpu(), real_B.squeeze(0).cpu(),real_C.squeeze(0).cpu(),
                                   fake_B1.squeeze(0).cpu(),fake_C5.squeeze(0).cpu(),\
                                    fake_B1toA.squeeze(0).cpu(), fake_C5toA.squeeze(0).cpu()], 0) 
        
        display_imgB = torch.stack([real_B.squeeze(0).cpu(), real_C.squeeze(0).cpu(),real_A.squeeze(0).cpu(),
                                   fake_A2.squeeze(0).cpu(),fake_C4.squeeze(0).cpu(),\
                                    fake_A2toB.squeeze(0).cpu(), fake_C4toB.squeeze(0).cpu()], 0) 

        display_imgC = torch.stack([real_C.squeeze(0).cpu(), real_A.squeeze(0).cpu(),real_B.squeeze(0).cpu(),
                                   fake_A6.squeeze(0).cpu(),fake_B3.squeeze(0).cpu(),\
                                    fake_A6toC.squeeze(0).cpu(), fake_B3toC.squeeze(0).cpu()], 0)
        
        display_img = torch.stack([real_A.squeeze(0).cpu(), real_B.squeeze(0).cpu(),real_C.squeeze(0).cpu(),
                                fake_B1.squeeze(0).cpu(),fake_C5.squeeze(0).cpu(),\
                                fake_B1toA.squeeze(0).cpu(), fake_C5toA.squeeze(0).cpu(),\
                                real_B.squeeze(0).cpu(), real_C.squeeze(0).cpu(),real_A.squeeze(0).cpu(),
                                fake_A2.squeeze(0).cpu(),fake_C4.squeeze(0).cpu(),\
                                fake_A2toB.squeeze(0).cpu(), fake_C4toB.squeeze(0).cpu(),\
                                real_C.squeeze(0).cpu(), real_A.squeeze(0).cpu(),real_B.squeeze(0).cpu(),
                                fake_A6.squeeze(0).cpu(),fake_B3.squeeze(0).cpu(),\
                                fake_A6toC.squeeze(0).cpu(), fake_B3toC.squeeze(0).cpu()], 0)
        
        # save_image(fake_B1, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-AtoB/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        # save_image(fake_C5, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-AtoC/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        
        # save_image(fake_A2, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-BtoA/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        # save_image(fake_C4, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-BtoC/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        
        # save_image(fake_A6, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-CtoA/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        # save_image(fake_B3, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-CtoB/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) #range= (0,128)
        
        
        save_image(display_imgA, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-display/A-Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False) 
        save_image(display_imgB, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-display/B-Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False)
        save_image(display_imgC, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-display/C-Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),normalize=False, scale_each=False)
        save_image(display_img, "../3cGAN_test_outputs/%s-%sep-%s-Est-Depths-display/Est-Depths-%s.png" % (opt.test_model, opt.epoch, opt.network_name, num),nrow=7, normalize=False, scale_each=False)

testing()


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

"""
import numpy as np
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2
#调用
mean_squared_error(y_test,y_predict)
mean_absolute_error(y_test,y_predict)
np.sqrt(mean_squared_error(y_test,y_predict))  # RMSE就是对MSE开方即可
r2_score(y_test,y_predict)
"""