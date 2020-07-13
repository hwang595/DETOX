from .utils import *
from .baseline_master import SyncReplicasMaster_NN
from functools import reduce

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class CodedMaster(SyncReplicasMaster_NN):
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._num_grad_to_collect = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._coded_grads_buffer = {}
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._update_mode = kwargs['update_mode']
        self._max_steps = kwargs['max_steps']
        self._group_list = kwargs['group_list']
        self._compress_grad = kwargs['compress_grad']
        self._device = kwargs['device']

        ## for speeding up maj vote stage of Draco ##
        self._model_shape_pointer = []
        #############################################

        self._group_size = len(self._group_list[0])

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
        elif self.network_config == "FC":
            self.network=FC_NN_Split()
        elif self.network_config == "logistic":
            self.network=LogisticRegression()
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn()
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG16":
            self.network=vgg16_bn(num_classes=100)

        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad)
        self.init_model_shapes()
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)

        self.network.to(self._device)

    def init_model_shapes(self):
        tmp_aggregate_buffer = []
        self._model_param_counter = 0
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            num_params = reduce((lambda x, y: x * y), shape)

            temp_tuple = (self._model_param_counter, self._model_param_counter+num_params)
            self._model_param_counter += num_params
            self._model_shape_pointer.append(temp_tuple)            

            self._model_shapes.append(shape)
            self._grad_aggregate_buffer.append(np.zeros(shape, dtype=np.float32))
            tmp_aggregate_buffer.append(np.zeros(shape))

        if self._update_mode == "maj_vote":
            for k, v in self._group_list.items():
                for i, l in enumerate(v):
                    if k not in self._coded_grads_buffer.keys():
                        #self._coded_grads_buffer[k] = [copy.deepcopy(tmp_aggregate_buffer)]
                        self._coded_grads_buffer[k] = [np.zeros(self._model_param_counter, dtype=np.float32)]
                    else:
                        #self._coded_grads_buffer[k].append(copy.deepcopy(tmp_aggregate_buffer))
                        self._coded_grads_buffer[k].append(np.zeros(self._model_param_counter, dtype=np.float32))

    def start(self):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure value fetched from ps is 1
        self.async_bcast_step()

        # fake test here:
        for i in range(1, self._max_steps+1):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            logger.info("Master node is entering step: {}".format(i))
            self.async_bcast_step()

            self.async_bcast_layer_weights_bcast()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()
            # wait for enough gradients to be aggregated:
            while not enough_gradients_received:
                status = MPI.Status()
                if self._compress_grad == "None":
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress":
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    received_grad=decompress(received_msg)

                if status.tag-88 in self.grad_accumulator.model_index_range:
                    if not self._first_grad_received:
                        self._first_grad_received=True
                        grad_gather_start_time = time.time()

                    layer_index = status.tag-88

                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])

                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(received_grad, layer_index, status.source)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)
            
            if self._update_mode == "normal":
                method_start = time.time()
                self._avg_received_grads()
                method_duration = time.time() - method_start
            elif self._update_mode == "maj_vote":
                # under development, stay tunned
                method_start = time.time()
                self._grad_majority_vote()
                method_duration = time.time() - method_start

            update_start = time.time()
            # update using SGD method
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)
            # update `state_dict` in pytorch modules
            #self.model_update(tmp_module)
            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()
            # save model for validation in a pre-specified frequency
            #if self.cur_step%self._eval_freq == 0:
            #    self._save_model(file_path=self._generate_model_path())
            logger.info("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            self.cur_step += 1

    def aggregate_gradient(self, gradient, layer_idx, source):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        if self._update_mode == "normal":
            self._grad_aggregate_buffer[layer_idx] += gradient
        elif self._update_mode == "maj_vote":
            _shape = gradient.shape
            for k, v in self._group_list.items():
                if source in v:
                    #assert self._coded_grads_buffer[k][v.index(source)][layer_idx].shape == gradient.shape
                    indices = self._model_shape_pointer[layer_idx]
                    if len(_shape) == 1:
                        self._coded_grads_buffer[k][v.index(source)][indices[0]:indices[1]] = gradient
                    elif len(_shape) > 1:
                        self._coded_grads_buffer[k][v.index(source)][indices[0]:indices[1]] = gradient.reshape((reduce(lambda x, y: x * y, _shape),))
                    #self._coded_grads_buffer[k][v.index(source)][layer_idx] = gradient

    def _grad_majority_vote(self):
        for k, v in self._coded_grads_buffer.items():
            #for j, _ in enumerate(self.network.parameters()):
            _maj_counter = 0
            for i, elem in enumerate(v):
                if _maj_counter == 0:
                    _maj_grad = elem
                    _maj_counter = 1
                elif np.array_equal(elem, _maj_grad):
                    _maj_counter += 1
                else:
                    _maj_counter -= 1
            # at the end of this loop we will find the majority grad
            index_pointer = 0
            for j, p in enumerate(self.network.parameters()):
                grad_size = reduce((lambda x, y: x * y), p.size())
                maj_vote_temp = _maj_grad[index_pointer:index_pointer+grad_size]
                index_pointer += grad_size
                self._grad_aggregate_buffer[j] += maj_vote_temp.reshape(self._grad_aggregate_buffer[j].shape)
            #assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
            #self._grad_aggregate_buffer[j] += _maj_grad

        self._grad_aggregate_buffer = [x / len(self._group_list) for x in self._grad_aggregate_buffer]

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape, dtype=np.float32)