from .utils import *
from .baseline_worker import DistributedWorker

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class CodedWorker(DistributedWorker):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._adversery = kwargs['adversery']
        self._err_mode = kwargs['err_mode']
        self._group_list = kwargs['group_list']
        self._train_dir = kwargs['train_dir']
        self._eval_freq = kwargs['eval_freq']
        self._max_steps = kwargs['max_steps']

        # only for test
        self._fail_workers = kwargs['adversaries']

        self._group_seeds = kwargs['group_seeds'] 
        self._group_num = kwargs['group_num'] # which group this worker belongs to
        self._group_size = len(self._group_list[0])
        self._compress_grad = kwargs['compress_grad']
        self._device = kwargs['device']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def build_model(self, input_size=784, num_classes=10):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18()
        elif self.network_config == "ResNet34":
            self.network=ResNet34()
        elif self.network_config == "ResNet50":
            self.network=ResNet50()
        elif self.network_config == "FC":
            self.network=FC_NN_Split()
        elif self.network_config == "logistic":
            self.network=LogisticRegression(input_size=input_size, num_classes=num_classes)
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn()
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG16":
            self.network=vgg16_bn(num_classes=100)

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()

        self.network.to(self._device)

    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1

        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        assert(self.cur_step == STEP_START_)

        # number of batches in one epoch
        num_batch_per_epoch = len(train_loader.dataset) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step = 0
        iter_start_time = 0
        first = True
        iter_avg_prec1 = 0
        iter_avg_prec5 = 0
        # use following flags to achieve letting each worker compute more batches
        should_enter_next = False

        logger.info("Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            # after each epoch we need to make sure workers in the same group re-shuffling using the same seed
            torch.manual_seed(self._group_seeds[self._group_num]+num_epoch)
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                X_batch, y_batch = train_image_batch.to(self._device), train_label_batch.to(self._device)
                while True:
                    # the worker shouldn't know the current global step except received the message from parameter server
                    self.async_fetch_step()
                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step()
                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue
                    # the real start point of this iteration
                    iter_start_time = time.time()
                    first = False
                    should_enter_next = False
                    logger.info("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

                    fetch_weight_start_time = time.time()
                    self.async_fetch_weights_bcast()
                    fetch_weight_duration = time.time() - fetch_weight_start_time

                    self.network.train()
                    self.optimizer.zero_grad()
                    # forward step
                    forward_start_time = time.time()
                    logits = self.network(X_batch)

                    loss = self.criterion(logits, y_batch)
                    forward_duration = time.time()-forward_start_time

                    # backward step
                    backward_start_time = time.time()
                    loss.backward()
                    backward_duration = time.time() - backward_start_time
                    computation_time = forward_duration + backward_duration

                    grads = [p.grad.detach().numpy().astype(np.float32) for p in self.network.parameters()]

                    prec1, prec5 = accuracy(logits.detach(), train_label_batch.long(), topk=(1, 5))
                    # in current setting each group cotains k workers, we let each worker calculate k same batches
                    c_start = time.time()
                    self._send_grads(grads)
                    c_duration = time.time() - c_start

                    logger.info('Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, Comp: {:.4f}, Comm: {:.4f}, Prec@1: {}, Prec@5: {}'.format(self.rank,
                         self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                            (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.item(), time.time()-iter_start_time, computation_time, c_duration+fetch_weight_duration, prec1.numpy()[0], prec5.numpy()[0]))
                    if self.cur_step%self._eval_freq == 0 and self.rank==1:
                        #self._save_model(file_path=self._generate_model_path())
                        self._save_model(file_path=self._generate_model_path())
                    break

    def _send_grads(self, grads):
        req_send_check = []
        #for i, grad in enumerate(reversed(grads)):
        for i, grad in enumerate(grads):
            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self.rank in self._fail_workers[self.cur_step]:
                simulation_grad = err_simulation(grad, self._err_mode)
                if self._compress_grad=='compress':
                    _compressed_grad = compress(simulation_grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                if self._compress_grad=='compress':
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
        req_send_check[-1].wait()