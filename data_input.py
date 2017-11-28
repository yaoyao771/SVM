#incoding:utf-8
import pickle
import numpy as np
import os
from scipy.misc import imread
def load_CIFAR_batch(filename):
    print(filename)
    with open(filename,'rb')as f:
        datadict=pickle.load(f,encoding='bytes')
        X=datadict[b'data']
        Y=datadict[b'labels']
        X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        Y=np.array(Y)
        return X,Y
def load_CIFAR10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(ROOT,'data_batch_%d'%(b,))
        X,Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr=np.concatenate(xs)
    Ytr=np.concatenate(ys)
    del X,Y
    Xte,Yte=load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr,Ytr,Xte,Yte
'''
def load_tiny_imagenet(path,dtpy=np.float32):
    with open(os.path.join(path,'wnids.txt'),'r')as f:
        wnids=[x.strip()for x in f]
    wnid_to_label={wnid:i for i,wnid in enumerate(wnids)}
    with open(os.path.join(path, 'wnids.txt'), 'r')as f:
        wnid_to_words=dict(line.split('\t')for line in f)
        for wnid,words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_name=[wnid_to_words[wnid]for wnid in wnids]

    X_train=[]
    y_train=[]
    for i,wnid in enumerate(wnids):
        if(i+1)%20==0:
            print ('loading training data for synset %d/%d'% (i+1,len(wnids)) )
        boxes_file=os.path.join(path,'train',wnid, '%s_boxes.text'% wnid)
        with open(boxes_file,'r')as f:
            filenames=[x.splot('\t')[0]for x in f]
        num_images=len(filenames)

        X_train_block=np.zeros((num_images,3,64,64),dtype=dtype)
        y_train_block=np.wnid_to_label[wnid]*np.ones(num_images,dtype=np.int64)
        for i,img_file in enumerate(filenames):
            img_file=os.path.join(path,'train',wnid,'images',img_file)
            img=imread(img_file)
            if img.ndim==2:
                img.shape=(64,64,1)
            X_train.append(X_train_block)
            y_train.append(y_train_block)
        X_train=np.concatenate(X_train,axis=0)
        y_train=np.concatenste(y_train,axis=0)

        with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
            img_files = []
            val_wnids = []
            for line in f:
                img_file, wnid = line.split('\t')[:2]
                img_files.append(img_file)
                val_wnids.append(wnid)
            num_val = len(img_files)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
            X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
            for i, img_file in enumerate(img_files):
                img_file = os.path.join(path, 'val', 'images', img_file)
                img = imread(img_file)
                if img.ndim == 2:
                    img.shape = (64, 64, 1)
                X_val[i] = img.transpose(2, 0, 1)

                # Next load test images
                # Students won't have test labels, so we need to iterate over files in the
                # images directory.
        img_files = os.listdir(os.path.join(path, 'test', 'images'))
        X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'test', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_test[i] = img.transpose(2, 0, 1)

        y_test = None
        y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
        if os.path.isfile(y_test_file):
            with open(y_test_file, 'r') as f:
                img_file_to_wnid = {}
                for line in f:
                    line = line.split('\t')
                    img_file_to_wnid[line[0]] = line[1]
            y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
            y_test = np.array(y_test)

        return class_names, X_train, y_train, X_val, y_val, X_test, y_test

'''
def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt) will
    be skipped.
    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.
    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models