import os
import json
import numpy as np
from torchvision import transforms
import json
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def compute_nDCG_by_sim(imgs, rcps, retrieved_range=1000, save_dir='./', where='fake'):
    """
    compute nDCG using real rank
    Arguments:
        imgs - image features
        rcps - recipe features
    Returns:
        mean of nDCG
        std of nDCG
    """
    N = retrieved_range
    data_size = imgs.shape[0]
    idxs = range(N)
    nDCGs = []
    # average over 10 sets
    plt.figure(figsize=(20, 6))
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        imgs_sub = imgs_sub / np.linalg.norm(imgs_sub, axis=1)[:, None]
        rcps_sub = rcps_sub / np.linalg.norm(rcps_sub, axis=1)[:, None]
        sims_i2r = np.dot(imgs_sub, rcps_sub.T) # [N, N]
        sims_r2r = np.dot(rcps_sub, rcps_sub.T)
        # loop through the N similarities for images
        for ii in idxs:
            # get a column of similarities for image ii
            sim_i2r = sims_i2r[ii,:]
            # sort indices in descending order
            sorting = np.argsort(sim_i2r)[::-1].tolist()
            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii) # the true recipe ranks at pos
            
            # DCG
            sim_r2r = sims_r2r[ii,:]
            sim_r2r_by_rank = sim_r2r[sorting]
            dominator = np.array([np.log2(i+1+1) for i in range(retrieved_range)])
            DCG = (sim_r2r_by_rank / dominator).sum()
            # if ii==0:
            #     plt.subplot(2,5,i+1)
            #     plt.hist(x=sim_r2r_by_rank[:pos+1], alpha=0.5, label='before')
            #     plt.hist(x=sim_r2r_by_rank[pos:], alpha=0.5, label='after')
            #     plt.text(23, 45, 'pos = {:.2f}'.format(pos))
            #     plt.legend()

            # ideal DCG
            sim_r2r_sorted = np.sort(sim_r2r)[::-1]
            iDCG = (sim_r2r_sorted / dominator).sum()

            if ii==0:
                plt.subplot(2,5,i+1)
                plt.plot(sim_r2r_by_rank / dominator, label='DCGs')
                plt.plot(sim_r2r_sorted / dominator, alpha=0.5, label='iDCGs')
                plt.title('pos = {:d}, DCG={:.2f}, iDCG={:.2f}'.format(pos, DCG, iDCG))
                plt.legend()

            nDCG = DCG / iDCG
            nDCGs.append(nDCG)

    # plt.savefig(os.path.join(save_dir, 'rel_by_sim_{}.png'.format(where)))
    plt.savefig(os.path.join(save_dir, 'DCGs_by_sim_{}.png'.format(where)))
    return np.array(nDCGs)


def compute_nDCG_by_bleu(img_feats, rcp_feats, rcps, retrieved_range=1000, save_dir='./', where='fake'):
    """
    compute nDCG using BLEU
    Arguments:
        img_feats - image features
        rcps - recipe features
        rcps - recipes
    Returns:
        mean of nDCG
        std of nDCG
    """
    N = retrieved_range
    data_size = img_feats.shape[0]
    idxs = range(N)
    cc = SmoothingFunction()
    nDCGs = []
    # average over 10 sets
    plt.figure(figsize=(20, 6))
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        img_feats_sub = img_feats[ids_sub, :]
        img_feats_sub = img_feats_sub / np.linalg.norm(img_feats_sub, axis=1)[:, None]
        rcp_feats_sub = rcp_feats[ids_sub, :]
        rcp_feats_sub = rcp_feats_sub / np.linalg.norm(rcp_feats_sub, axis=1)[:, None]
        sims_i2r = np.dot(img_feats_sub, rcp_feats_sub.T) # [N, N]

        rcps_sub = [rcps[j] for j in ids_sub] 
        # loop through the N similarities for images
        for ii in tqdm(idxs):
            # get a column of similarities for image ii
            sim_i2r = sims_i2r[ii,:]
            # sort indices in descending order
            sorting = np.argsort(sim_i2r)[::-1].tolist()
            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii) # the true recipe ranks at pos
            # DCG
            # sort recipes 
            rcps_sub_by_rank = [tpl[0] for tpl in sorted(zip(rcps, sorting), key=lambda x:-x[1])]
            # get candidate
            candidate = (' '.join(rcps_sub[ii]['instructions'])).split()
            bleus = []
            for jj in sorting:
                reference = (' '.join(rcps_sub[jj]['instructions'])).split()
                bleu = sentence_bleu([reference], candidate, smoothing_function=cc.method4)
                bleus.append(bleu)
            bleus = np.array(bleus)
            dominator = np.array([np.log2(i+1+1) for i in range(retrieved_range)])
            DCG = (bleus / dominator).sum()
            # if ii==0:
            #     plt.subplot(2,5,i+1)
            #     plt.hist(x=bleus[:pos+1], alpha=0.5, label='before')
            #     plt.hist(x=bleus[pos:], alpha=0.5, label='after')
            #     plt.text(23, 45, 'pos = {:.2f}'.format(pos))
            #     plt.legend()

            # ideal DCG
            bleus_sorted = np.sort(bleus)[::-1]
            iDCG = (bleus_sorted / dominator).sum()

            if i==0 and ii<10:
                plt.subplot(2,5,ii+1)
                plt.plot(bleus / dominator, label='DCGs')
                plt.plot(bleus_sorted / dominator, alpha=0.5, label='iDCGs')
                plt.title('pos = {:d}, DCG={:.2f}, iDCG={:.2f}'.format(pos, DCG, iDCG))
                plt.legend()

            plt.savefig(os.path.join(save_dir, 'DCGs_by_bleu_{}.png'.format(where)))
            if ii==9:
                break

            nDCG = DCG / iDCG
            nDCGs.append(nDCG)

        break
    # plt.savefig(os.path.join(save_dir, 'rel_by_bleu_{}.png'.format(where)))
    return np.array(nDCGs)


def compute_average_relevance_by_sim(imgs, rcps, K=10, retrieved_range=1000, save_dir='', where='fake'):
    """
    compute nDCG using real rank
    Arguments:
        imgs - image features
        rcps - recipe features
    Returns:
        mean of nDCG
        std of nDCG
    """
    N = retrieved_range
    data_size = imgs.shape[0]
    idxs = range(N)
    # create 10 sets
    avgs = []
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        imgs_sub = imgs_sub / np.linalg.norm(imgs_sub, axis=1)[:, None]
        rcps_sub = rcps_sub / np.linalg.norm(rcps_sub, axis=1)[:, None]
        sims_i2r = np.dot(imgs_sub, rcps_sub.T) # [N, N]
        sims_r2r = np.dot(rcps_sub, rcps_sub.T)
        # loop through the N similarities for images
        avg_topK_sub = []
        for ii in idxs:
            # get a column of similarities for image ii
            sim_i2r = sims_i2r[ii,:]
            # sort indices in descending order
            sorting = np.argsort(sim_i2r)[::-1].tolist()
            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii) # the true recipe ranks at pos
            
            # DCG
            sim_r2r = sims_r2r[ii,:]
            sim_r2r_by_rank = sim_r2r[sorting]
            avg_topK = np.mean(sim_r2r_by_rank[:K])
            avg_topK_sub.append(avg_topK)
        avgs.append(np.mean(avg_topK_sub))
    
    plt.hist(x=avgs, alpha=0.5, label=where)
    plt.savefig(os.path.join(save_dir, 'average_relevance_by_sim_{}.png'.format(where)))
    return np.array(avgs)

if __name__ == '__main__':
    imgs = np.random.rand(3000, 1000)
    rcps = np.random.rand(3000, 1000)
    compute_nDCG_by_sim(imgs, rcps)