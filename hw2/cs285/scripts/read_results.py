import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    currentpatn = os.path.abspath(__file__)
    rootpath = os.path.dirname(currentpatn)
    basedir = os.path.dirname(os.path.dirname(rootpath))

    sub_title='q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_31-08-2021_22-04-10'
    logdir = basedir+'/data/'+sub_title+'/*'
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    plt.plot(X,Y)
    plt.title(sub_title)
    plt.show()
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))