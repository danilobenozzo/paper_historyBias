import pickle
from plotting_functions import figure_layout
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

if __name__=='__main__':
    
    label_name = ['samediff', 'diffsame']
    figure_label = ['d', 'c']
    i_cols = [[0, 2], [2, 0]]
    pwd_saved =  '%s/data/' % os.getcwd()
    pwd_saved_null_dist =  '%s/data/' % os.getcwd()
    filename_null_dist = 'results_decoding_setRandomStateNew_globalNullDist_onlyPositiveLabel_noGridS_current_trial_S1_3classes_tws_m400S11400_length200_step25_min10trialPerClass_localnTest_400trialPerClass_1000rep_area1_high1Hz'
    pwd_tosave_p = '%s/pvals/' % os.getcwd()
    pwd_tosave = '%s/figures/' % os.getcwd()

    for i_file, file_i in enumerate(label_name):
        
        filename_decoding = 'results_decoding_setRandomStateNew_noGridS_lSVC_current_trial_S1_%sColorS2oneBackS1Current_3classes_tws_m400S11400_square_length200_step25_min10trialPerClass_localnTest_400trialPerClass_100rep_area1_high1Hz' % file_i 
        
        dataset_decoding = pickle.load(open('%s%s.pickle' % (pwd_saved, filename_decoding)))
        result_decoding = np.array(dataset_decoding['result'], float)
        window_bin = dataset_decoding['window_bin']
        length = dataset_decoding['length']
        mean_acc_decoding = np.diagonal(np.mean(np.diagonal(result_decoding, axis1=-1, axis2=-2), -1), axis1=-1, axis2=-2)
        
        mean_acc_decoding_train = np.squeeze(mean_acc_decoding[:, 0])
        mean_acc_decoding_test = np.squeeze(mean_acc_decoding[:, 1])
        
        
        dataset = pickle.load(open('%s%s.pickle' % (pwd_saved_null_dist, filename_null_dist)))
        result_null_dist = np.array(dataset['result'], float)
        mean_null_dist = np.mean(np.diagonal(result_null_dist, axis1=-1, axis2=-2), -1)
        
        m1 = np.repeat(np.mean(mean_null_dist, 0), mean_null_dist.shape[0], axis=0).reshape([mean_null_dist.shape[1], mean_null_dist.shape[0]]).T
        k = np.std(mean_acc_decoding_train, 0) / np.std(mean_null_dist, 0)
        k = np.repeat(k, mean_null_dist.shape[0], axis=0).reshape([mean_null_dist.shape[1], mean_null_dist.shape[0]]).T
        mean_null_dist = m1 + (mean_null_dist-m1)*k
        
        p_val_null = np.zeros(mean_acc_decoding_train.shape[-1])
        for i, acc_i in enumerate(np.mean(mean_acc_decoding_train, 0)):
            p_val_null[i] = max(1.0/mean_null_dist.shape[0], ((mean_null_dist[:,i] > acc_i).sum()+1) / (float(mean_null_dist.shape[0])+1))
        
        #plotting
        x_in = 3
        y_in = 2
        fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize = figure_layout(x_in, 2.5)
        fig, ax = plt.subplots(figsize=(x_in, y_in))
        cmap = plt.get_cmap("tab20")     
        
        #null dist
        plt.plot(window_bin + length/2, np.mean(mean_null_dist, 0), '-', linewidth=linewidth_main, color='k', label='chance level')
        plt.plot(window_bin + length/2, np.mean(mean_null_dist, 0) + np.std(mean_null_dist, 0), '-', linewidth=linewidth, color='k')
        plt.plot(window_bin + length/2, np.mean(mean_null_dist, 0) - np.std(mean_null_dist, 0), '-', linewidth=linewidth, color='k')
        #########
        
        i_col = i_cols[i_file][0]
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_train, 0), '-', linewidth=linewidth_main, color=cmap.colors[i_col])#'r') cmap.colors[i_col]
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_train, 0) + np.std(mean_acc_decoding_train, 0), '-', linewidth=linewidth, color=cmap.colors[i_col])
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_train, 0) - np.std(mean_acc_decoding_train, 0), '-', linewidth=linewidth, color=cmap.colors[i_col])
        
        i_col = i_cols[i_file][1]
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_test, 0), '-', linewidth=linewidth_main, color=cmap.colors[i_col])#'r') cmap.colors[i_col]
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_test, 0) + np.std(mean_acc_decoding_test, 0), '-', linewidth=linewidth, color=cmap.colors[i_col])
        plt.plot(window_bin + length/2, np.mean(mean_acc_decoding_test, 0) - np.std(mean_acc_decoding_test, 0), '-', linewidth=linewidth, color=cmap.colors[i_col])
        
        p_val = np.zeros(mean_acc_decoding.shape[-1])
        for i_time, time_i in enumerate(window_bin):
            t, p_val[i_time] = stats.mannwhitneyu(mean_acc_decoding_train[:,i_time], mean_acc_decoding_test[:,i_time])#, equal_var=False)
        
        ax.fill_between(window_bin+length/2, np.ones(len(window_bin))*.2, np.ones(len(window_bin))*.225, where=np.logical_and(p_val<.05/len(window_bin), p_val_null<.05), facecolor='k' , alpha=.2)
        
        plt.xlim([window_bin[0], window_bin[-1] + length])
        plt.ylim([.2, .7])#([.2, 1.05])
        plt.xticks(np.arange(0,1001,500), np.arange(0,1001,500), fontsize=fontsize)
        plt.yticks(np.arange(.2, .75, .1), np.around(np.arange(.2, .75, .1), 1), fontsize=fontsize)
        plt.xlabel('time [ms]', fontsize=fontsize_main)
        plt.ylabel('acc', fontsize=fontsize_main)
        fig.tight_layout()
        plt.show()
        
        filename_figure = 'fig3%s' % figure_label[i_file]
        plt.savefig('%s%s.svg' % (pwd_tosave, filename_figure), format='svg')
        
        np.savetxt('%s%s.txt' % (pwd_tosave_p, filename_figure), p_val_null)
        
        ##########################
        # decoding maps
        dataset_decoding = pickle.load(open('%s%s.pickle' % (pwd_saved, filename_decoding)))
        result_decoding = np.array(dataset_decoding['result'], float)
        window_bin = dataset_decoding['window_bin']
        length = dataset_decoding['length']
        mean_acc_decoding = np.mean(np.diagonal(result_decoding, axis1=-1, axis2=-2), -1)
        
        for i_test in range(2):
            
            mean_acc_decoding_i = mean_acc_decoding[:,i_test,:,:]
            p_val = np.zeros(np.mean(mean_acc_decoding_i, 0).shape)
            for i in range(p_val.shape[0]):
                for j in range(p_val.shape[1]):
                    acc_ij = np.mean(mean_acc_decoding_i[:, i, j])
                    p_val[i, j] = max(1.0/mean_null_dist.shape[0], ((mean_null_dist[:,i] > acc_ij).sum()+1) / (float(mean_null_dist.shape[0])+1))
            
            #plotting
            x_in = 2.5
            y_in = 2
            fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize = figure_layout(x_in, 2.5)
            fig, ax = plt.subplots(figsize=(x_in, y_in))
            mtx_toplot = np.mean(mean_acc_decoding_i, 0)
            mtx_toplot[p_val > .05] = np.nan
            im = plt.imshow(mtx_toplot)
            
            time_bin = window_bin + length/2
            bin_i = range(12, 53, 20)
            plt.xticks(bin_i, time_bin[bin_i], fontsize=fontsize)
            plt.yticks(bin_i, time_bin[bin_i], fontsize=fontsize)
            plt.clim([1/3., .7])
            
            cbar = fig.colorbar(im, ticks = np.arange(.4, .71, .1), fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(np.arange(.4, .71, .1), rotation=0, fontsize=fontsize)
            plt.grid(linewidth=linewidth)
            
            fig.tight_layout()
            plt.show()
            filename_figure = 'fig3%s_%smap' % (figure_label[i_file], i_test)
            plt.savefig('%s%s.svg' % (pwd_tosave, filename_figure), format='svg')
            
            np.savetxt('%s%s.txt' % (pwd_tosave_p, filename_figure), p_val)
        ################################