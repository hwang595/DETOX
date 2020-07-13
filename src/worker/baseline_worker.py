from .utils import *

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DistributedWorker(NN_Trainer):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
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
        self._compress_grad = kwargs['compress_grad']
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._max_steps = kwargs['max_steps']
        self._update_mode = kwargs['update_mode']

        self._lis_simulation = kwargs['lis_simulation']

        # only for test      
        #self._fail_workers = [self.world_size-i for i in range(1, kwargs['worker_fail']+1)]
        self._fail_workers = kwargs['adversaries']

        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18()
        elif self.network_config == "ResNet34":
            self.network=ResNet34()
        elif self.network_config == "ResNet50":
            self.network=ResNet50()
        elif self.network_config == "ResNet101":
            self.network=ResNet101()
        elif self.network_config == "ResNet152":
            self.network=ResNet152()
        elif self.network_config == "FC":
            self.network=FC_NN()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn(num_classes=100)
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG16":
            self.network=vgg16_bn(num_classes=100)
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()

        #if self._checkpoint_step != 0:
        #    file_path = "../checkpoints/geo_median/model_step_"+str(self._checkpoint_step)
        #    self._load_model(file_path)

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()

    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        global STEP_START_

        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        if self._checkpoint_step == 0:
            assert(self.cur_step == STEP_START_)
        else:
            assert(self.cur_step == int(self._checkpoint_step)+1)

        # number of batches in one epoch
        num_batch_per_epoch = len(train_loader.dataset) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step=0
        iter_start_time=0
        first = True

        logger.info("Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                X_batch, y_batch = Variable(train_image_batch), Variable(train_label_batch)
                while True:
                    # the worker shouldn't know the current global step
                    # except received the message from parameter server
                    self.async_fetch_step()

                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step()

                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue

                    # the real start point of this iteration
                    iteration_last_step = time.time() - iter_start_time
                    iter_start_time = time.time()
                    first = False
                    logger.info("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

                    # TODO(hwang): return layer request here and do weight before the forward step begins, rather than implement
                    # the wait() in the fetch function
                    fetch_weight_start_time = time.time()
                    if self.comm_type == "Bcast":
                        self.async_fetch_weights_bcast()
                    elif self.comm_type == "Async":
                        self.async_fetch_weights_async()
                    fetch_weight_duration = time.time() - fetch_weight_start_time

                    # switch to training mode
                    self.network.train()
                    # manage batch index manually
                    self.optimizer.zero_grad()
                    # forward step
                    comp_start_time = time.time()
                    logits = self.network(X_batch)
                    loss = self.criterion(logits, y_batch)
                    loss.backward()
                    computation_time = time.time() - comp_start_time

                    grads = [p.grad.detach().numpy().astype(np.float32) for p in self.network.parameters()]
                    
                    # on the end of a certain iteration
                    prec1, prec5 = accuracy(logits.detach(), train_label_batch.long(), topk=(1, 5))

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

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def sync_fetch_step(self):
        '''fetch the first step from the parameter server'''
        self.next_step = self.comm.recv(source=0, tag=10)

    def async_fetch_step(self):
        req = self.comm.irecv(source=0, tag=10)
        self.next_step = req.wait()

    def async_fetch_weights_async(self):
        request_layers = []
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], source=0, tag=11+layer_idx)
                request_layers.append(req)

        assert (len(layers_to_update) == len(request_layers))
        weights_to_update = []
        for req_idx, req_l in enumerate(request_layers):
            req_l.wait()
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            # we also need to update the layer cur step here:
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)
    
    def async_fetch_weights_bcast(self):
        layers_to_update = []
        weights_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                weights_recv=self.comm.bcast(self.model_recv_buf.recv_buf[layer_idx], root=0)
                weights = w_decompress(weights_recv)
                weights_to_update.append(weights)
                self.model_recv_buf.layer_cur_step[layer_idx] = self.cur_step
        self.model_update(weights_to_update)
    
    def update_step(self):
        '''update local (global) step on worker'''
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name or "num_batches_tracked" in key_name:
                tmp_dict={key_name: param}
            else:
                assert param.size() == weights_to_update[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(weights_to_update[model_counter_])}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def _send_grads(self, grads):
        req_send_check = []
        for i, grad in enumerate(grads):
            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self._lis_simulation == "simulate":
                if self._compress_grad=='compress':
                    if self._update_mode == "sign-sgd":
                        # signSGD worker side
                        grad = np.sign(grad).astype(np.int8)
                        
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                if self.rank in self._fail_workers[self.cur_step]:
                    simulation_grad = err_simulation(grad, self._err_mode)
                    
                    if self._update_mode == "sign-sgd":
                        # signSGD worker side
                        simulation_grad = np.sign(simulation_grad).astype(np.int8)

                    if self._compress_grad=='compress':
                        _compressed_grad = compress(simulation_grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
                else:
                    if self._compress_grad=='compress':
                        if self._update_mode == "sign-sgd":
                            # signSGD worker side
                            grad = np.sign(grad).astype(np.int8)
                        
                        _compressed_grad = compress(grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
        req_send_check[-1].wait()

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = Variable(data, volatile=True), Variable(y_batch)
            output = self.network(data)
            #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).data[0]
            
            prec1_tmp, prec5_tmp = accuracy(output.data, target.data, topk=(1, 5))
            prec1_counter_ += prec1_tmp.numpy()[0]
            prec5_counter_ += prec5_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        logger.info('Test set: Step: {}, Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(self.cur_step, test_loss, prec1, prec5))

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            #torch.save(self.network, f_)
            torch.save(self.network.state_dict(), f_)
        return

    def _load_model(self, file_path):
        model_state_dict=torch.load(file_path)
        self.network.load_state_dict(model_state_dict)
        logger.info("Validation Worker Done Loading Checkpoint from {}".format(file_path))