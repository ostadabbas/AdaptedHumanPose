import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import glob
import datetime
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('whitegrid')

def tsne_plot(folder, output_file_path, max_num_samples_per_dataset, scaling=[0.5, 0.5, 0.5], sz_font=22):
    '''
    folder: "vis/train"
    output_file_path: /path/to/plot.png
    max_num_samples_per_dataset: 
    scaling: scaling factors. 
        example. if features are (2048, 8, 8) and scaling is (0.5, 0.75, 0.25),
        then features become (1024, 6, 2)
    '''
    print('==== [{}] IMPORTANT.. GENERATING TEST PLOT TO {} TO VERIFY VALID DESTINATION BEFORE GOING THROUGH COMPUTATIONS'.format(
        datetime.datetime.now(), output_file_path))
    sns.scatterplot(x=[1,2], y=[1,2]).get_figure().savefig(output_file_path)
    print('==== [{}] Output figure path validated. Continuing with calculations.'.format(datetime.datetime.now()))

    datasets = os.listdir(folder)  # ['Human36M', ...]

    # Load data
    all_files = []
    labels = []

    print('==== [{}] Loading files from {} datasets'.format(datetime.datetime.now(), len(datasets)))

    for dataset in datasets:
        feature_folder = os.path.join(folder, dataset, "G_fts_raw")
        numpy_files = glob.glob(os.path.join(feature_folder, "*npy"))
        np.random.shuffle(numpy_files)
        for file in tqdm(numpy_files[:max_num_samples_per_dataset], desc=dataset):
            x = np.load(file)
            assert x.shape == (2048, 8, 8)
            all_files.append(x)
            labels.append(dataset)

    np.save('labels.npy', labels)

    print('==== [{}] Done loading files. Loaded {} samples.'.format(datetime.datetime.now(), len(all_files)))

    # Reshape
    print('==== [{}] Downsampling features'.format(datetime.datetime.now()))
    all_files = zoom(all_files, (1,) + tuple(scaling))
    print('==== [{}] Done downsampling. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_files)))

    print('==== [{}] Reshaping feature array'.format(datetime.datetime.now()))
    new_shape = (len(all_files), np.prod(np.shape(all_files)[1:]))
    all_files = np.reshape(all_files, new_shape).astype(float)
    print('==== [{}] Done reshaping. Current shape: {}'.format(datetime.datetime.now(), np.shape(all_files)))

    # Run t-SNE
    print('==== [{}] Running t-SNE'.format(datetime.datetime.now()))
    model = TSNE(n_components=2)
    output = model.fit_transform(all_files)

    print('SAVED')
    np.save('output.npy', output)

    # Plot
    print('==== [{}] Plotting and saving figure'.format(datetime.datetime.now()))
    snsplot = sns.scatterplot(x=output[:, 0], y=output[:, 1], hue=labels, alpha=0.7)
    snsplot.get_figure().savefig(output_file_path, dpi=300)
    print('==== [{}] Figure saved to {}.'.format(datetime.datetime.now(), output_file_path))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='vis/train')
    parser.add_argument('-o', '--output')
    parser.add_argument('-m', '--max-num', type=int)
    parser.add_argument('-s1', '--scaling-one', type=float, default=0.5)
    parser.add_argument('-s2', '--scaling-two', type=float, default=0.5)
    parser.add_argument('-s3', '--scaling-three', type=float, default=0.5)
    args = parser.parse_args()
    tsne_plot(args.folder, args.output, args.max_num, [args.scaling_one, args.scaling_two, args.scaling_three])
