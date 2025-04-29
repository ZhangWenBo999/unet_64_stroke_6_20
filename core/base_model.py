import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn

# ***************** 新添加的包 ***************
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

import core.util as Util
CustomResult = collections.namedtuple('CustomResult', 'name result')

class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.opt = opt
        self.phase = opt['phase']
        self.set_device = partial(Util.set_device, rank=opt['global_rank'])

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']
        self.epoch = 0
        self.iter = 0 

        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        train_mse_losses = []
        eval_mae_losses = []
        while self.epoch < self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1
            if self.opt['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 
            self.opt["datasets"]["train"]["which_dataset"]["args"]["mask_config"]["phase"] = "train"
            train_log = self.train_step()
            train_mse_losses.append(train_log['train/mse_loss'])
            ''' save logged informations into log dict ''' 
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
                # self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
                path = self.opt['path']['checkpoint'] + '/last'

                if os.path.exists(path):
                    shutil.rmtree(path)
                last_path = os.path.join(self.opt['path']['checkpoint'], 'last')
                os.makedirs(last_path, exist_ok=True)
                self.save_everything()

            # if self.epoch % self.opt['train']['val_epoch'] == 0:

            if self.epoch > self.opt['train']['val_epoch']:
                # self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    self.opt["datasets"]["train"]["which_dataset"]["args"]["mask_config"]["phase"] = "eval"
                    val_log = self.val_step()
                    eval_mae_losses.append(val_log['val/FID'].to('cpu').tolist())
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value.item()))
                    # 保存最好的checkpoint
                    if val_log['val/FID'] < self.opt['train']['min_val_FID_loss']:
                        self.opt['train']['min_val_flag'] = True
                        path = self.opt['path']['checkpoint'] + '/best'

                        if os.path.exists(path):
                            shutil.rmtree(path)
                        best_path = os.path.join(self.opt['path']['checkpoint'], 'best')
                        os.makedirs(best_path, exist_ok=True)

                        self.opt['train']['min_val_FID_loss'] = val_log['val/FID']
                        self.epoch = str(self.epoch) + '_best'
                        self.save_everything()
                        self.epoch = int(self.epoch.split('_', 1)[0])

                    # 到达指定轮数保存训练和验证损失图
                    if self.epoch == self.opt['train']['n_epoch']:
                        self.opt['train']['train_previous'].extend(train_mse_losses)
                        self.opt['train']['eval_previous'].extend(eval_mae_losses)
                        pic_path = os.path.join(self.opt['path']['checkpoint'], 'loss_pic')
                        os.makedirs(pic_path, exist_ok=True)
                        # 到达指定轮数，保存checkpoint 并 画图
                        plt.figure()
                        plt.title('Train Curve')

                        # 去除顶部和右边的框
                        ax = plt.gca()
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')

                        # 设置x和y轴的标签
                        plt.xlabel('epochs')
                        plt.ylabel('train_losses')

                        epochs = np.arange(len(self.opt['train']['train_previous']))
                        plt.plot(epochs, self.opt['train']['train_previous'])

                        # 保存训练损失图
                        save_path = os.path.join(self.opt['path']['checkpoint'], 'loss_pic', 'train_losses.png')
                        plt.savefig(save_path)

                        plt.figure()
                        plt.title('Eval Curve')

                        # 去除顶部和右边的框
                        ax = plt.gca()
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')

                        # 设置x和y轴的标签
                        plt.xlabel('epochs')
                        plt.ylabel('eval_losses')

                        epochs = np.arange(len(self.opt['train']['eval_previous']))
                        plt.plot(epochs, self.opt['train']['eval_previous'])

                        # 保存验证损失图
                        save_path = os.path.join(self.opt['path']['checkpoint'], 'loss_pic', 'eval_losses.png')
                        plt.savefig(save_path)

                        plt.show()
                        self.logger.info('train_last: {}\t'.format(self.opt['train']['train_previous']))
                        self.logger.info('eval_last: {}\t'.format(self.opt['train']['eval_previous']))
            # self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        if self.opt['train']['min_val_flag']:
            save_filename = '{}_{}.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], 'best', save_filename)
        else:
            save_filename = '{}_{}.pth'.format(self.epoch, network_label)
            save_path = os.path.join(self.opt['path']['checkpoint'], 'last', save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state'] is None:
            return 
        self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self. opt['path']['resume_state'], network_label)
        
        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: Util.set_device(storage)), strict=strict)

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        if self.opt['global_rank'] !=0:
            return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        if self.opt['train']['min_val_flag']:
            self.opt['train']['min_val_flag'] = False
            save_filename = '{}.state'.format(self.epoch)
            save_path = os.path.join(self.opt['path']['checkpoint'], 'best', save_filename)
        else:
            save_filename = '{}.state'.format(self.epoch)
            save_path = os.path.join(self.opt['path']['checkpoint'], 'last', save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self. opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self. opt['path']['resume_state'])
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')
