import logging
import os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "28"
import cv2
from logger import logger
import mxnet as mx
import numpy as np
import sys

sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time
from os.path import join
logging.basicConfig(level=logging.DEBUG)

def all_in_one(class_id,num_id,lr_sch_size,epoch_num,new_train=True,pretrain_model_epoch = 0,random_crop=True,lr_rate=0.005):
    resnet_version = '152'
    attr_len = [int(num_id)]
    mu = 2
    width = 110 * mu
    longth = int(160/8*7 * mu)
    longth_ = int(longth/8*6)
    width_ = int(width /8*6)
    batch_size = 96 *int(((2/mu)**2))
    augs = mx.image.CreateAugmenter(data_shape=(3,longth_, width_),rand_crop=random_crop,rand_resize=random_crop, rand_mirror=True, brightness=0.125, contrast=0.125, rand_gray=0.125,saturation=0.125,hue=0.125, pca_noise=0.05, inter_method=10)
    augs = [mx.image.CenterCropAug(size=(width,longth))] + augs
    
    augs_val = [mx.image.CenterCropAug(size=(width,longth)),mx.image.ForceResizeAug((width_,longth_))]
    if not random_crop:
        augs = augs_val
    #folder_name = 'first_class_data/'
    folder_name = '{}_data_{}/'.format(class_id,num_id)
    checkpoint_name ='fine-tuned-resnet{}-{}-{}'.format(resnet_version,class_id,num_id) 
    if new_train:
        pretrain_model_name = 'resnet-{}'.format(resnet_version)
    else:
        pretrain_model_name = join(folder_name,checkpoint_name)
    #pretrain_model_name = checkpoint_name
    begin_epoch = pretrain_model_epoch
    train_iter = mx.image.ImageIter(batch_size=batch_size, data_shape=(3,longth_,width_), label_width=1,
                                   path_imgidx=folder_name+'cloth_train.idx', path_imgrec=folder_name+'cloth_train.rec', shuffle=True,
                                   aug_list=augs)
    val_iter= mx.image.ImageIter(batch_size=batch_size, data_shape=(3, longth_, width_), label_width=1,
                                   path_imgidx=folder_name+'cloth_val.idx', path_imgrec=folder_name+'cloth_val.rec', shuffle=False,
                                   aug_list=augs_val)
    train =train_iter
    val= val_iter
    #val = None

    def get_fine_tune_model(symbol, arg_params, layer_name='flatten0',new_train=True):
        """
        symbol: the pretrained network symbol
        arg_params: the argument parameters of the pretrained model
        num_classes: the number of classes for the fine-tune datasets
        layer_name: the layer name before the last fully-connected layer
        """
        if not new_train:
            logger.info('continue train')
            return (symbol,arg_params)
        all_layers = symbol.get_internals()
        net = all_layers[layer_name + '_output']
        multi_output = []
        for idx, i in enumerate(attr_len):
            net_temp = mx.symbol.FullyConnected(data=net, num_hidden=i, name='fc{}'.format(idx))
            net_temp = mx.symbol.SoftmaxOutput(data=net_temp, name='softmax'.format(idx))
            multi_output.append(net_temp)
        multi_output = mx.sym.Group(multi_output)
        def judge(k):
            for i in range(len(attr_len)):
                if 'fc{}'.format(i) in k:
                    return True
            return False
        new_args = dict({k: arg_params[k] for k in arg_params if not judge(k)})
        return (multi_output, new_args)

    sym, arg_params, aux_params = mx.model.load_checkpoint(pretrain_model_name, pretrain_model_epoch)

    (new_sym, new_args) = get_fine_tune_model(sym, arg_params,new_train=new_train)

    lr_sch = mx.lr_scheduler.FactorScheduler(step=int((lr_sch_size)/batch_size), factor=0.95,stop_factor_lr=1e-5)
    lr_rate = lr_rate*(0.95**pretrain_model_epoch)
    sgd = mx.optimizer.SGD(momentum=0.9, multi_precision=False,lr_scheduler=lr_sch,learning_rate=lr_rate)
    logger.info(lr_rate)
    sgd.rescale_grad = 1/batch_size
    #print(new_sym.list_arguments())
    temp = []
    for i in new_sym.list_arguments():
        if i.find('stage3') != -1:
            break
        temp.append(i)
    d = {}
    for i in temp:
        d[i] = 1
    sgd.set_lr_mult(d)
    def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
        devs = [mx.gpu(i) for i in range(num_gpus)]
        mod = mx.mod.Module(symbol=symbol, context=devs,label_names=['softmax_label'])
        mod.fit(train, val,
                num_epoch=epoch_num,
                arg_params= arg_params,
                aux_params= aux_params,
                allow_missing=True,
                batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                optimizer = sgd,
                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
                eval_metric=[mx.metric.TopKAccuracy(top_k=3),mx.metric.TopKAccuracy(top_k=2),'acc'],#Multi_Accuracy(num=len(attr_len)),
                begin_epoch=begin_epoch, #get 30
                epoch_end_callback=mx.callback.do_checkpoint(join(folder_name,checkpoint_name), 5))


    def parse_train_and_eval():
        num_gpus = 2
        fit(new_sym, new_args, aux_params,train,val, batch_size, num_gpus)

    print('folder_name:',folder_name,resnet_version,class_id,num_id,pretrain_model_name,pretrain_model_epoch,checkpoint_name)
    print('begin_epoch',begin_epoch)
    print('lr_sch_size',lr_sch_size)
    print(attr_len)
    parse_train_and_eval()
#all_in_one(class_id='23',num_id='7',lr_sch_size=150000,epoch_num=30)
#all_in_one(class_id='25',num_id='35',lr_sch_size=200000,epoch_num=10,pretrain_model_epoch=0,new_train=True,random_crop=True)
#all_in_one(class_id='21',num_id='23',lr_sch_size=200000,epoch_num=20,pretrain_model_epoch=10,new_train=False,random_crop=True)
#all_in_one(class_id='46',num_id='4',lr_sch_size=15000,epoch_num=20)
#all_in_one(class_id='33',num_id='3',lr_sch_size=15000,epoch_num=30,pretrain_model_epoch=10,new_train=False)
all_in_one(class_id='33',num_id='3',lr_sch_size=15000,epoch_num=30)
