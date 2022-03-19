import torch
import numpy as np
import cv2
from network import CCNN
from torch.autograd import Variable
import os
from tensorboardX import SummaryWriter
from util.loss import similar_loss
from torchvision import transforms
import util.data as utils
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
torch.cuda.set_device(0)


def feed_random_seed(seed=np.random.randint(1, 10000)):
    #    feed random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_latest_file_name(dir):
    lists = os.listdir(dir)
    if len(lists) > 0:
        lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn))
        file_new = os.path.join(dir, lists[-1])
        return file_new
    return None


def get_model(model_name):
    if model_name == "CCNN":
        return CCNN.Net().cuda()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(epoch, dataloader, net, optimizer, alpha, m=0):
    accum_loss = 0
    net.train()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for i, data in enumerate(dataloader):
        images, labels, _ = Variable(data['images']), Variable(
            data['labels']), data['paths']
        shape = list(images.size())
        images = images.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        images, labels = images.cuda(), labels.cuda()
        b, fea = net(images)
        prediction = torch.argmax(b, 1)
        correct += (prediction == labels).sum().float()
        total += len(labels)
        loss = similar_loss(b, labels, fea, alpha, m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum_loss += loss.data
        print(f'[{epoch}][{i}/{len(dataloader)}] loss: {loss.data:.4f} ',
              end="\r", flush=False)

    acc_str = ((correct/total).cpu().detach().data.numpy())
    return accum_loss/len(dataloader), acc_str


def test(epoch, dataloader, net, alpha, m=0):
    accum_loss = 0
    net.eval()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels, _ = Variable(data['images']), Variable(
                data['labels']), data['paths']
            shape = list(images.size())
            images = images.reshape(shape[0] * shape[1], *shape[2:])
            labels = labels.reshape(-1)
            images, labels = images.cuda(), labels.cuda()
            b, fea = net(images)
            prediction = torch.argmax(b, 1)
            correct += (prediction == labels).sum().float()
            total += len(labels)
            loss = similar_loss(b, labels, fea, alpha, m)
            accum_loss += loss.data
            print(f'[{epoch}][{i}/{len(dataloader)}] test_loss: {loss.data:.4f} ',
                  end="\r", flush=False)
    acc_str = ((correct/total).cpu().detach().data.numpy())
    return accum_loss / len(dataloader), acc_str


def main(args):
    resume_epoch = args.resume_epoch
    niter = args.niter
    model_checkpoints = args.model_checkpoints
    m = args.m
    alpha = args.alpha
    mode = args.mode

    output_root = "output/"+args.model_name+"/A_S_256_0.4_pair/"
    train_cover_dir = args.dataset+"/train/cover"
    train_stego_dir = args.dataset+"/train/stego"
    valid_cover_dir = args.dataset+"/val/cover"
    valid_stego_dir = args.dataset+"/val/stego"
    test_cover_dir = args.dataset+"/test/cover"
    test_stego_dir = args.dataset+"/test/stego"

    os.makedirs(output_root, exist_ok=True)
    feed_random_seed()
    train_transform = transforms.Compose([
        utils.ToTensor(),
    ])

    # load datasets
    train_loader = utils.DataLoaderStego(train_cover_dir, train_stego_dir,
                                         embedding_otf=False, shuffle=True,
                                         pair_constraint=True,
                                         batch_size=args.batch_size,
                                         transform=train_transform,
                                         )

    val_loader = utils.DataLoaderStego(valid_cover_dir, valid_stego_dir,
                                       embedding_otf=False, shuffle=False,
                                       pair_constraint=True,
                                       batch_size=args.batch_size,
                                       transform=train_transform,
                                       )
    test_loader = utils.DataLoaderStego(test_cover_dir, test_stego_dir,
                                        embedding_otf=False, shuffle=False,
                                        pair_constraint=True,
                                        batch_size=args.batch_size,
                                        transform=train_transform,
                                        )

    model = get_model(args.model_name)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.95, eps=1e-8,
                                     weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.2)

    outf = os.path.join(output_root, "model")
    os.makedirs(outf, exist_ok=True)

#   setting file
    setting_str = f"setup config:\n\tdataset:{args.dataset}\n\t"\
        f"net :{args.model_name} \n\tniter:{niter} \n\tis_scheduler:{args.is_schedular} \n\t"\
        f"batch size:{args.batch_size}\n\tlearning rate:{args.learning_rate}\n\tis_logger:{args.is_logger}\n\t"\
        f"alhpa:{alpha}\n\tmodel_checkpoint:{args.model_checkpoints}\n\t"\
        f"auto_train:{args.auto_train}\n\toutput root:{output_root}\n\tm:{m}\n\t"
    print(setting_str)

    with open(output_root+"/setting.txt", "w") as f:
        f.writelines(setting_str)

    if args.is_logger:
        logger = None

    if mode == "test":
        weights_file = output_root+"/model/best_model.pth"
        checkpoint = torch.load(weights_file, map_location='cuda:0')
        model.load_state_dict(checkpoint['net'])
        model.eval()
    elif args.auto_train and args.pretrained_model == "":
        weights_file = get_latest_file_name(output_root+"/model")
        if weights_file is not None:
            checkpoint = torch.load(weights_file)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    # 		swtich optimizer coefficient to cuda type
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            if args.is_scheduler is True:
                scheduler.load_state_dict(checkpoint['scheduler'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            resume_epoch = checkpoint['epoch'] + 1
            print(
                f"auto_weight:successfully find latest weight:{weights_file} and load,resume epoch changes to {resume_epoch}")
    elif args.auto_train:
        ckpt = torch.load(args.pretrained_model)
        model.load_state_dict(ckpt['net'])

    best_acc = 0

    if mode == "test":
        test_loss, test_acc = test(0, test_loader, model, alpha, m)
        print(f'test loss:{test_loss:.4f}test_acc:{test_acc}')
    elif mode == "train":
        for epoch in range(resume_epoch, niter + 1):
            train_loss, train_acc = train(
                epoch, train_loader, model, optimizer, alpha, m)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss, test_acc = test(epoch, val_loader, model, alpha, m)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            learning_rate_list.append(get_lr(optimizer))
            if args.is_scheduler is True:
                scheduler.step(epoch=epoch)
            torch.cuda.empty_cache()
            if args.is_logger:
                if logger is None:
                    logger_root = "runs/"
                    if not output_root is None:
                        logger_root = os.path.join(output_root, logger_root)
                    logger = SummaryWriter(logger_root)
                if epoch % model_checkpoints == 0:
                    check_loss, check_acc = test(
                        epoch, test_loader, model, alpha)
                    print(f'test loss:{check_loss:.4f} test_acc:{ check_acc}')
                    logger.add_scalar('test_loss', check_loss, epoch + 1)
                    logger.add_scalar('test_acc', check_acc, epoch + 1)
    #               log out
                    for inx in range(model_checkpoints):
                        history_epoch = epoch - model_checkpoints + inx
                        print(train_acc_list[inx])
                        logger.add_scalar(
                            'train_loss', train_loss_list[inx], history_epoch + 1)
                        logger.add_scalar(
                            'val_loss', test_loss_list[inx], history_epoch + 1)
                        logger.add_scalar(
                            'learning rate', learning_rate_list[inx], history_epoch + 1)
                        logger.add_scalar(
                            'train_acc', train_acc_list[inx], history_epoch + 1)
                        logger.add_scalar(
                            'val_acc', test_acc_list[inx], history_epoch + 1)
                    train_loss_list = []
                    test_loss_list = []
                    learning_rate_list = []
                    train_acc_list = []
                    test_acc_list = []
                    if args.is_scheduler is True:
                        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'scheduler': scheduler.state_dict(),
                                 'epoch': epoch}
                    else:
                        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'epoch': epoch}
                    torch.save(state, os.path.join(outf, f'{epoch:04d}.pth'))
                    if check_acc > best_acc:
                        torch.save(state, os.path.join(
                            outf, f'best_model.pth'))
                        best_acc = check_acc

                print(f'[{epoch}] train loss: {train_loss:.4f} val loss:{test_loss:.4f} lr:{get_lr(optimizer)} train_acc:{train_acc} val_acc:{test_acc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=2,
        help="Model definition file."
    )

    parser.add_argument(
        "--pretrained_model",
        default='',
        help="Trained model weights file."
    )

    parser.add_argument(
        "--model_name",
        default="CCNN",
        help="Choose which model to train."
    )

    parser.add_argument(
        "--auto_train",
        default=True,
        type=bool,
        help="Continue training."
    )

    parser.add_argument(
        "--resume_epoch",
        default=1,
        type=int,
        help="Resume epoch."
    )

    parser.add_argument(
        "--niter",
        default=400,
        type=int,
        help="The number of epochs."
    )

    parser.add_argument(
        "--is_schedular",
        default=True,
        type=bool,
        help="Dynamic learning."
    )

    parser.add_argument(
        "--is_logger",
        default=True,
        type=bool,
        help="Log file."

    )

    parser.add_argument(
        "--model_checkpoints",
        type=int,
        default='10',
        help="Checkpoints."
    )

    parser.add_argument(
        "--dataset",
        default='dataset/A_S_256_0.4_pair',
        help="Data folder."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default='0.4',
        help="Initial learning rate."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default='0.05',
        help="The parameter of loss."
    )

    parser.add_argument(
        "--m",
        type=float,
        default='3',
        help="Distance of pair-wise."
    )

    parser.add_argument(
        "--mode",
        default='test',
        help="Train or test."
    )

    args = parser.parse_args()

    main(args)
