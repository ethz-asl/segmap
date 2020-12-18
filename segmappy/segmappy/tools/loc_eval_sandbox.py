import numpy as np
import matplotlib.pyplot as plt

def main():
    print('Evaluate localization quality.')
    loc_file_dir = '/media/nikhilesh/Nikhilesh/SemSegMap/Bags/NCLT/paper_Eval/localization_eval/'
    loc_file_dir_name = loc_file_dir + 'config4d.txt'
    print('File name: ' + loc_file_dir_name)

    # Load data.
    with open(loc_file_dir_name) as f:
        content = f.readlines()
    
    num_loc = len(content)
    print('Number of localizations: ' + str(num_loc))

    # Compute some stats.
    loc_data = None
    i = 0
    for entry in content:
        if loc_data is None:
            loc_data = np.fromstring(entry, sep=',')
        else:
            loc_data = np.vstack((loc_data,np.fromstring(entry, sep=',')))
    print('DONE loading!')
    print('Avg error x: ' + str(np.mean(loc_data[:,1])))
    print('Avg error y: ' + str(np.mean(loc_data[:,2])))
    print('Avg error z: ' + str(np.mean(loc_data[:,3])))

    # Plot loc vs error (x = index of localization, y = absolute error), does it go down?
    err_x = loc_data[:,1]
    err_y = loc_data[:,2]
    err_z = loc_data[:,3]
    abs_err = np.sqrt(err_x*err_x + err_y*err_y + err_z*err_z)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # plot the cumulative histogram
    # plt.plot(np.arange(abs_err.shape[0]),abs_err)
    n, bins, patches = plt.hist(abs_err, 100, density=True, histtype='step', cumulative=True)
    print('Eyoo')
    print(n)
    print(bins)
    # print(patches)
    # print(np.where(n<0.61)[0][-1])
    # print(bins[28])
    # print(n.shape)
    threshold = 0.6
    err = bins[np.where(n <= threshold)[0][-1]]
    plt.hlines(threshold, bins[0], err)
    plt.vlines(err, 0, threshold)
    print(str(threshold*100)+' percent of LOC are within: ' + str(err))
    plt.text(1, 1.0, 'Num of Loc: ' + str(num_loc) + ' ' + str(100*threshold) + ' perc below: ' + str(err))

    # ToDo(alaturn) Mark 60%, annotate # LOC
    plt.grid(True)
    plt.legend(loc='right')
    plt.title('Cumulative     Localization Error')
    plt.xlabel('Absolute error (m)')
    plt.ylabel('Cumulative')
    # plt.show()
    plt.savefig(loc_file_dir_name.split(".")[0] + "_cdf.png", facecolor='w', edgecolor='w')



if __name__ == "__main__":
    main()

