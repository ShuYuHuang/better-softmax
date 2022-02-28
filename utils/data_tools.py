import numpy as np
from functools import partial

import tensorflow as tf
from tensorflow import keras


###  Class-Specific Loader  ###
def class_slices_loader(mixed_loader,classes_list,cache=True):
    dataset_list=[]
    for yy in classes_list:
        if cache: 
            dataset_list.append(mixed_loader.filter(lambda x,y: y==yy).cache().shuffle(2000))
        else:
            dataset_list.append(mixed_loader.filter(lambda x,y: y==yy).shuffle(2000))
    return dataset_list

###  Meta Loader for Paired Data  ###
def paired_meta_generator(dataset_list,mult=40):
    orders=np.concatenate([np.random.permutation(len(dataset_list)) for x in range(mult)])
    for tasks in range(len(orders)//2):
        picked=[dataset_list[tt] for tt in orders[2*tasks:2*(tasks+1)]]
        # 選一個人當 support'''
        idx_s=np.random.choice(range(2), size=1, replace=False)
        # 選1張照片當 query'''
        idx_q=np.random.choice(range(2), size=1, replace=False)
        if idx_q[0]==idx_s[0]:
            tmp=next(iter(picked[idx_s[0]].batch(2)))[0]
            support=tmp[-1]
            query=tmp[0]
        else:
            support=next(iter(picked[idx_s[0]]))[0]
            query=next(iter(picked[idx_q[0]]))[0]
        x=tf.stack((support,query), axis=0)
        # 如果該query照片跟support同種就是1，不同就是0'''
        y=tf.constant(idx_s==idx_q,dtype=tf.float32)
        yield x, y

###  Meta Loader for Multiple Groups  ###
def multiclass_meta_generator(dataset_list,mult=40,ways=5,shots=3):
    orders=np.concatenate([np.random.permutation(len(dataset_list)) for x in range(mult)])
    for tasks in range(len(orders)//ways):
        #從已決定好的順序拉出WAY個class#
        picked=[dataset_list[tt] for tt in orders[ways*tasks:ways*(tasks+1)]]
        #每個class抽shots+1張#
        data = [next(iter(class_loader.batch(shots+1)))[0] for class_loader in picked]
        #support每個class各有shots張照片#
        support=tf.concat([d[:-1] for d in data],axis=0)
        #query挑每個class 1 張，順序不固定#
        idxs=np.random.choice(range(ways), size=ways, replace=False)
        query=tf.concat([data[wayid][-2:-1] for wayid in idxs],axis=0)
        #輸出的時候把support跟query接在一起#
        yield tf.concat([support, query], axis=0), tf.stack([keras.utils.to_categorical(idx,num_classes=ways) for idx in idxs], axis=0)
        
###  Convinient Function for Generating Loaders  ###        
def form_meta_loader(loader,cfg):
    
    source_class_loader=class_slices_loader(loader,cfg.source_classes)
    val_class_loader=class_slices_loader(loader,cfg.val_classes)
#     target_class_loader=class_slices_loader(train_loader,cfg.target_classes)
    
    # For loading data pairs used in paired network such as Siamese network
    if cfg.meta_type=="paired":
        meta_train_loader=tf.data.Dataset.from_generator(
            partial(paired_meta_generator,source_class_loader,cfg.mult),
            output_types=(tf.float32,tf.float32),
            output_shapes=((2,cfg.w,cfg.h,cfg.ch),1))\
                .cache()\
                .shuffle(1000)\
                .prefetch(cfg.meta_batch_size)\
                .batch(cfg.meta_batch_size)
        meta_val_loader=tf.data.Dataset.from_generator(
            partial(paired_meta_generator,val_class_loader,cfg.mult),
            output_types=(tf.float32,tf.float32),
            output_shapes=((2,cfg.w,cfg.h,cfg.ch),1))\
                .cache()\
                .prefetch(cfg.meta_batch_size*4)\
                .batch(cfg.meta_batch_size*4)
                
    # For loading data groups used in multi-class networks such as Prototypical networks
    if cfg.meta_type=="multiclass":    
        meta_train_loader=tf.data.Dataset.from_generator(
            partial(multiclass_meta_generator,source_class_loader,cfg.mult,cfg.ways,cfg.shots),
            output_types=(tf.float32,tf.float32),
            output_shapes=((cfg.ways*(cfg.shots+1),cfg.w,cfg.h,cfg.ch),1))\
                .cache()\
                .shuffle(1000)\
                .prefetch(cfg.meta_batch_size)\
                .batch(cfg.meta_batch_size)
        meta_val_loader=tf.data.Dataset.from_generator(
            partial(multiclass_meta_generator,val_class_loader,cfg.mult,cfg.ways,cfg.shots),
            output_types=(tf.float32,tf.float32),
            output_shapes=((cfg.ways*(cfg.shots+1),cfg.w,cfg.h,cfg.ch),1))\
                .cache()\
                .prefetch(cfg.meta_batch_size*4)\
                .batch(cfg.meta_batch_size*4)
    return meta_train_loader,meta_val_loader,source_class_loader,val_class_loader