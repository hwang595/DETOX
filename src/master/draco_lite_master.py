from .utils import *
from .baseline_master import SyncReplicasMaster_NN

import logging
import torch.optim as optim

from joblib import Parallel, delayed
from functools import reduce

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DracoLiteMaster(SyncReplicasMaster_NN):
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
        self.num_workers = self.world_size-1
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
        self._group_size = len(self._group_list[0])
        self._bucket_size = kwargs['bucket_size']
        self._lis_simulation = kwargs['lis_simulation']
        self._s = kwargs['worker_fail']
        self._device = kwargs['device']

        ######## LR scheduling related ############
        self.gamma = 0.99
        self.lr_step = 100000000000000000
        ###########################################

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
            self.network=FC_NN()
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn(num_classes=100)
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG19":
            self.network=vgg19_bn(num_classes=100)

        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad)
        self.init_model_shapes()
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        #self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

        self.network.to(self._device)

    def init_model_shapes(self):
        tmp_aggregate_buffer = []
        self._model_param_counter = 0
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            num_params = reduce((lambda x, y: x * y), shape)
            self._model_param_counter += num_params

            self._model_shapes.append(shape)
            self._grad_aggregate_buffer.append(np.zeros(shape, dtype=np.float32))
            tmp_aggregate_buffer.append(np.zeros(shape, dtype=np.float32))

        #if self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        for k, v in self._group_list.items():
            for i, l in enumerate(v):
                if k not in self._coded_grads_buffer.keys():
                    self._coded_grads_buffer[k] = [copy.deepcopy(tmp_aggregate_buffer)]
                else:
                    self._coded_grads_buffer[k].append(copy.deepcopy(tmp_aggregate_buffer))

        # buffer setted up for draco-lite aggregations
        self._sub_grad_size = int(self.num_workers/self._group_size)
        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=np.float32)

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


            ################### "A Little is enough" attack simulation on the PS"#########################
            # TODO (hongyi & shashank): try to see how to make this practical 
            if self._lis_simulation == "simulate":
                self._LIE_attack_simulation()
            else:
                pass            

            method_start = time.time()
            #self._grad_aggregate_buffer = draco_lite_aggregation(self._coded_grads_buffer, self._bucket_size, self.network, self._grad_aggregate_buffer)
            self._draco_lite_aggregation()
            method_duration = time.time() - method_start

            update_start = time.time()
            # update using SGD method
            self._model_update()

            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            logger.info("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            
            #for param_group in self.optimizer.param_groups:
            #    break
            logger.info("Real time Lr: {}".format([param_group['lr'] for param_group in self.optimizer.param_groups]))   

            self.cur_step += 1

    def aggregate_gradient(self, gradient, layer_idx, source):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        #if self._update_mode == "normal":
        #    self._grad_aggregate_buffer[layer_idx] += gradient
        #elif self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        # under development, stay tunned
        for k, v in self._group_list.items():
            if source in v:
                assert self._coded_grads_buffer[k][v.index(source)][layer_idx].shape == gradient.shape
                self._coded_grads_buffer[k][v.index(source)][layer_idx] = gradient

    def _model_update(self):
        # we implement a simple lr scheduler here. TODO (hwang): see if there exists a better method to fit into PyTorch lr_scheduler
        if self.cur_step % self.lr_step == 0:
            self.optimizer.lr_update(updated_lr=(self.lr * self.gamma ** (self.cur_step // self.lr_step)))
        self.optimizer.step(grads=self._grad_aggregate_buffer, mode="draco_lite")

    def _get_geo_median(self, bucket_grads):
        geo_median = np.array(hd.geomedian(np.array(bucket_grads), axis=0))
        return geo_median

    # elementwise maj vote among gradients gathered by PS
    def _elemwise_median(self, bucket_grads):
        elem_median = np.median(np.array(bucket_grads), axis=0)
        return elem_median

    # single-core version
    def _draco_lite_aggregation_single_thread(self):
        majority_grads = self._grad_majority_vote()
        # n-layers, r-groups: then majority should be in r * n:
        for j, p in enumerate(self.network.parameters()):
            layer_majority_grads = np.array([mg[j] for mg in majority_grads])
            indices = np.arange(len(layer_majority_grads))
            np.random.shuffle(indices)
            random_indicies = np.split(indices, self._bucket_size)
            for buckets in random_indicies:
                bucket_grads = np.take(layer_majority_grads, buckets, axis=0)
                #geo_median = self._get_geo_median(bucket_grads)
                elem_median = self._elemwise_median(bucket_grads)
                self._grad_aggregate_buffer[j] += elem_median.reshape(p.size())
        self._grad_aggregate_buffer = [x/float(len(self._group_list)) for x in self._grad_aggregate_buffer]

    # multi-core optimized verion
    def _draco_lite_aggregation(self):
        self._grad_majority_vote()

        # optimization objectives:
        # i) get rid of concated_grads and making it predefined
        # ii) use fancy slicing instead of the second for loop to get rid of grad_transformer (done, need to be further tested)
        # need to double check
        if self._update_mode == "coord-median":
            # bucketing stage
            indices = np.arange(self._sub_grad_size)
            np.random.shuffle(indices)
            random_indicies = np.array(np.split(indices, self._bucket_size))

            grad_transformer = self._draco_lite_aggregation_buffer[random_indicies, :]
            num_buckets = grad_transformer.shape[0]

            # aggregation stage
            grad_mean = np.mean(grad_transformer, axis=1)
            aggr_res = np.median(grad_mean, axis=0)
        elif self._update_mode == "bulyan":
            def __bulyan(grad_list, nb_in_score, bulyan_inds):
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = []
                for i, g_i in enumerate(grad_list):
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j:
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)

                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind)))

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                selected_grads = np.array(grad_list)[topk_ind, :]

                # starting the second stage bulyan step
                # let's start to parallelize this step: the strategy is parallel([nb_in_score, d/k], ..., [nb_in_score, d/k])
                grad_dim = selected_grads.shape[1]

                temp_selected_grads = []
                num_pieces = 8
                segment_size = int(grad_dim / num_pieces)
                sement_counter = 0
                for i in range(num_pieces-1):
                    temp_selected_grads.append(selected_grads[:, sement_counter:sement_counter+segment_size])
                    sement_counter += segment_size
                temp_selected_grads.append(selected_grads[:, sement_counter:])

                temp_sorted_selected_grads = Parallel(n_jobs=-1, prefer="threads")(delayed(np.sort)(g, axis=0) for g in temp_selected_grads)
                sorted_selected_grads = np.concatenate(temp_sorted_selected_grads, axis=1)
                
                bulyan_selected_grads = sorted_selected_grads[bulyan_inds, :]
                aggregated_grad = np.mean(bulyan_selected_grads, axis=0)
                return aggregated_grad

            # figure out bulyan statistics
            # where `effective_s` denotes the worst case number of Byzantine workers after the majority voting stage
            effective_s = math.floor(self._s / math.ceil(self._group_size/2))
            nb_in_score = self._draco_lite_aggregation_buffer.shape[0] - effective_s - 2
            pivot = int(nb_in_score/2)
            bulyan_s = nb_in_score - 2 * effective_s

            if nb_in_score % 2 == 0:
                # even case
                bulyan_inds = [(pivot- 1 - i) for i in range(1, int((bulyan_s-2)/2)+1)] + [pivot-1, pivot] + [(pivot + i) for i in range(1, int((bulyan_s-2)/2)+1)]
            else:
                # odd case
                bulyan_inds = [(pivot - i) for i in range(1, int((bulyan_s-1)/2)+1)] + [pivot] + [(pivot + i) for i in range(1, int((bulyan_s-1)/2)+1)]

            aggr_res = __bulyan(self._draco_lite_aggregation_buffer, nb_in_score, bulyan_inds)

        elif self._update_mode == "multi-krum":
            def __krum(grad_list, s, num_workers):
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = []
                for i, g_i in enumerate(grad_list):
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j:
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)

                # compute scores
                nb_in_score = num_workers - s - 2
                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind)))

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                aggregated_grad = np.mean(np.array(grad_list)[topk_ind, :], axis=0)
                return aggregated_grad

            effective_s = math.floor(self._s / math.ceil(self._group_size/2))
            
            # hard coded for now
            #temp_aggr_res = []
            #for i in range(2):
            #    temp_grads = grad_transformer[i]
            #    temp = __krum(temp_grads, effective_s, temp_grads.shape[0])
            #    temp_aggr_res.append(temp)
            #aggr_res_temp = np.concatenate(temp_aggr_res, axis=0)

            # with bucketing:
            # bucketing stage in multi-krum backened draco-lite
            indices = np.arange(self._sub_grad_size)
            np.random.shuffle(indices)

            ## Note that this part is hard coded currently
            random_indicies = np.array_split(indices, 2)
            grad_transformer = np.array([self._draco_lite_aggregation_buffer[rand_inds, :] for rand_inds in random_indicies])
            aggr_res_temp = Parallel(n_jobs=-1, prefer="threads")(delayed(__krum)(grad_transformer[i], effective_s, grad_transformer[i].shape[0]) for i in range(2))
            aggr_res = np.mean(np.array(aggr_res_temp), axis=0)

            # without bucketing:
            #aggr_res = __krum(self._draco_lite_aggregation_buffer, effective_s, self._draco_lite_aggregation_buffer.shape[0])

        index_pointer = 0
        for j, p in enumerate(self.network.parameters()):
            grad_size = reduce((lambda x, y: x * y), p.size())
            drac_lite_median = aggr_res[index_pointer:index_pointer+grad_size]
            index_pointer += grad_size
            self._grad_aggregate_buffer[j] = drac_lite_median.reshape(self._grad_aggregate_buffer[j].shape)


    def _grad_majority_vote(self):
        for k, v in self._coded_grads_buffer.items():
            index_pointer = 0
            for j, p in enumerate(self.network.parameters()):
                grad_size = reduce((lambda x, y: x * y), p.size())
                _maj_counter = 0

                for i, elem in enumerate(v):
                    if _maj_counter == 0:
                        _maj_grad = elem[j]
                        _maj_counter = 1
                    elif np.array_equal(elem[j], _maj_grad):
                        _maj_counter += 1
                    else:
                        _maj_counter -= 1
                try:
                    assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                except AssertionError as e:
                    warnings.warn("Gradient shapes incompatible, deprecated! ")

                self._draco_lite_aggregation_buffer[k, index_pointer:index_pointer+grad_size] = _maj_grad.reshape(-1)
                index_pointer += grad_size

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape, dtype=np.float32)

        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=np.float32)

    def _LIE_attack_simulation(self):
        """
        Simulating the attack method in: https://arxiv.org/abs/1902.06156
        (Otheriwse, 2-round communication strategy needs to be implemented for each iteration)
        The process is farily simple: (p_mal)_i = mu_i + z^{max} * simga_i, for i in {1, ..., d}
        """
        self.__z = 1.0
        dummpy_adversarial_nodes = np.random.choice(self.num_workers, self._s, replace=False)
        
        for j, _ in enumerate(self.network.parameters()):
            # aggregate coded gradients from different groups together, this is stupid, try to avoid this
            tempt_grads = []
            for k, v in self._coded_grads_buffer.items():
                for i, elem in enumerate(v):
                    tempt_grads.append(elem[j])
            mu, sigma = np.mean(tempt_grads, axis=0), np.std(tempt_grads, axis=0)

            for adv_index in dummpy_adversarial_nodes:
                for k, v in self._coded_grads_buffer.items():
                    if adv_index in self._group_list[k]:
                        _mal_grad = mu + self.__z * sigma
                        _relative_index_in_group = self._group_list[k].index(adv_index)
                        assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                        self._coded_grads_buffer[k][_relative_index_in_group][j] =  mu + self.__z * sigma