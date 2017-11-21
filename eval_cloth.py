import mxnet as mx
import cv2
import numpy as np
from os.path import join
import os
from logger import logger
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
def eval_res(class_id,num_id,eval_epoch,root_dir = '/home/lhw/face/faceRec'):
    print(class_id,num_id,eval_epoch)
    batch_size = 512
    val_label = None
    imgs = None

    attr_len = [int(num_id)]
    mu = 2
    width = 110 * mu
    longth = int(160/8*7 * mu)
    longth_ = int(longth/8*6)
    width_ = int(width /8*6)
    folder_name = '{}_data_{}/'.format(class_id,num_id)
    class_num_id = '{}_data_{}/'.format(class_id,num_id)
    resnet_version = '152'
    checkpoint_name ='fine-tuned-resnet{}-{}-{}'.format(resnet_version,class_id,num_id) 
    pretrain_model_name = 'resnet-{}'.format(resnet_version)


    def ge_val_label(folder_name):
        val_label = None
        augs = mx.image.CreateAugmenter(data_shape=(3,longth_, width_),rand_crop=True,rand_resize=False, rand_mirror=True, brightness=0.125, contrast=0.125, rand_gray=0.05,saturation=0.125, pca_noise=0, inter_method=10)
        val_iter= mx.image.ImageIter(batch_size=1, data_shape=(3, longth_, width_), label_width=1,
                                       path_imgidx='cloth_val.idx', path_imgrec='cloth_val.rec', shuffle=False,
                                       aug_list=augs)
        for batch in val_iter:
            l = batch.label[0]
            if val_label is None:
                val_label = l
            else:
                val_label = mx.ndarray.concat(val_label,l,dim=0)
            #print(mx.ndarray.concat(l,l,dim=1))
        val_label = val_label.as_in_context(mx.cpu())
        print(val_label.shape)
        return val_label

    def eval_avage(fea_list,val_label):
        res_final = mx.nd.zeros(fea_list[0].shape,ctx=mx.gpu())
        val_label = val_label.as_in_context(mx.gpu())
        #res_final = final_res[0]
        for i in fea_list:
            res_final += i
        res_final/= len(fea_list)
        acc = mx.metric.Accuracy()
        top2 = mx.metric.TopKAccuracy(top_k=2)
        acc.update(labels=[val_label],preds=[res_final])
        top2.update(labels=[val_label],preds=[res_final])
        logger.info(acc.get())
        logger.info(top2.get())
        return res_final
    os.chdir(join(root_dir,folder_name))
    def do_multi_predict(model_name,batch_size=512,folder_name=folder_name,epoch_num=eval_epoch,tt=1):
        #sym, arg_params, aux_params = mx.model.load_checkpoint('fine-tuned-firstclass-res18-use-resize', 25)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name,epoch_num)
        mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,longth_,width_))], 
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)
        augs = [mx.image.CenterCropAug(size=(width,longth)),mx.image.ForceResizeAug((width_,longth_))]
        #augs += mx.image.CreateAugmenter(data_shape=(3,longth_, width_),rand_crop=False,rand_resize=False, rand_mirror=True, brightness=0.125, contrast=0.125, rand_gray=0.05,saturation=0.125, pca_noise=0, inter_method=10)
        final_res = None
        print(model_name,epoch_num)
        # define a simple data batch
        for i in range(tt):
            print(i)
            val_iter= mx.image.ImageIter(batch_size=batch_size, data_shape=(3, longth_, width_), label_width=1,
                                           path_imgidx='cloth_val.idx', path_imgrec='cloth_val.rec', shuffle=False,
                                           aug_list=augs)
            res = mod.predict(val_iter,always_output_list=True)
            if final_res is None:
                final_res = res
            else:
                final_res += res
        del mod
        return final_res
    fea_list = do_multi_predict(checkpoint_name,tt=5,epoch_num=eval_epoch)

    #res_final = final_res[0]
    '''
    res_final = mx.nd.zeros(fea_list[0].shape,ctx=mx.cpu())
    for i in fea_list:
        res_final += i
    res_final/= len(fea_list)
    '''
    #print(1,10,3,4,5,6,7,8)
    #rrrr =  ['帽子','鞋子','披带类','上装','裤子','裙子','连体装','包包']
    #print(res_final,rrrr[res_final[0].asnumpy().argmax()])
    val_label= ge_val_label(folder_name=folder_name)

    res_final= eval_avage(fea_list[0:5],val_label)
    p = res_final.asnumpy()
    l = val_label.asnumpy()
    pred = []
    ff = []
    tt = []
    fff = []
    ttf = []
    ttc = []
    allf = []
    the = 0.90
    for i in range(len(p)):
        pp = p[i]
        mm = pp.argmax()
        if mm != l[i]:
            allf.append(pp[mm])
        if mm != l[i] and pp[mm] <the:
            ff.append(pp[mm])
        else:
            if pp[mm] < the:
                tt.append(pp[mm])
                ttc.append(mm)
    #logger.info(len(tt),len(ff),len(ttf),len(fff),len(allf))
    logger.info((len(tt)+len(ff))/res_final.shape[0])
    logger.info((len(allf)-len(ff))/res_final.shape[0])
    preds = res_final.asnumpy()
    preds = preds.argmax(axis=1)
    preds.shape
    from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,accuracy_score
    cm = confusion_matrix(l,preds)
    cm = (cm/cm.sum(axis=1)[:,np.newaxis])
    from pprint import pprint
    logger.info(cm)
    print(cm)
    #print(len([i for i in ttc if i==])/500)
eval_res(10001,3,10,'/root/cloth')
#eval_res(23,7,10,'/root/cloth')
