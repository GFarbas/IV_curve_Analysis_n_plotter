from matplotlib import pyplot as plt
import pandas as pd



def plot_iv(title, x_title, y_title, x, y):
    # Plot the points
    try:
        curve_labels = title
        if y[50] < -10:  #if the measurement at index 50 is below -10mA/cm2
            plt.plot(x, y, label=curve_labels, alpha=0.8)
            plt.ylabel(y_title)
            plt.axis([-0.4, 1, -45, 50])
        else:
            plt.semilogy(x, y, label=curve_labels, alpha=0.8)
            plt.ylabel('ln_'+ y_title)
            plt.axis([0, 0.9, -45, 50])
        plt.title(title)
        plt.xlabel(x_title)
        plt.axvline(x=0, label=curve_labels[1], linestyle="solid", color="black",linewidth=1)
        plt.axhline(y=0, label=curve_labels[1], linestyle="solid", color="black",linewidth=1)
    except:
        print("All IV curves ----------------")
        for i in range(len(x)):
            plt.plot(x[i], y[i], label=curve_labels[i], linestyle="-", alpha=0.5, linewidth=1)
        plt.legend(fontsize=8)
        # plt.plot(x, y, label=curve_labels,fontsize=8, alpha=0.8)
        plt.title("all IV curve")
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current Density [mA/cm2]")
        plt.axvline(x=0, label=curve_labels[1], linestyle="solid", color="black",linewidth=1)
        plt.axhline(y=0, label=curve_labels[1], linestyle="solid", color="black",linewidth=1)
        plt.axis([-0.4, 1, -45, 50])


def plot_EQE_graph(title, x_title,y_title,x,y):
    # Plot the points
    try:
        curve_labels= title
        plt.plot(x, y, label=curve_labels,alpha=0.8)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

    except:
        print("All EQEs----------------")
        for i in range(len(x)):
            plt.plot(x[i], y[i],label=curve_labels[i], linestyle="-", alpha=0.8,linewidth=2)
        plt.legend(fontsize=8)
        #plt.plot(x, y, label=curve_labels,fontsize=8, alpha=0.8)
        plt.title("All EQEs")
        plt.xlabel("nm")
        plt.ylabel("EQE")
        plt.axis([250, 1400, 0, 1])

    if save_plots==1:
        plt.savefig('plt_' + title + '.png')
    else:
        plt.show()

def plot_EQE_graphs(title, x_title,y_title,x1,y1,x2=0,y2=0,point1=0,point2=0):
    try:
        int(x2)
        x_values = x1
        y_values = y1
        curve_labels = ['Measurement']
        plt.plot(x_values, y_values, label=curve_labels[0], alpha=0.8)
        plt.axis([250, 1400, 0, 1])
        if point1 != 0:
            curve_labels = ['"Eg_1= ' + str(point1) + ' nm ']
            plt.axvline(x=point1, label=curve_labels[0], linestyle="dashed", color="gray", alpha=0.7)
            if point2 != 0:
                curve_labels = ['"Eg_1= ' + str(point1) + ' nm ', ' Eg_2= ' + str(point2) + 'nm ']
                plt.axvline(x=point2, label=curve_labels[1], linestyle="dashed", color="gray", alpha=0.3)
    except:
        print(f"{x2} is not an integer")
        x_values = [x1, x2]
        y_values = [y1, y2]
        curve_labels = ['1st derivative', 'Derivative from interpo...']
        plt.plot(x_values[0], y_values[0], label=curve_labels[0], alpha=0.8)
        plt.plot(x_values[1], y_values[1], label=curve_labels[1], linestyle = "dashed",alpha=0.8)
        plt.axvline(x=point1, label=curve_labels[1], linestyle="dashed", color="gray", alpha=0.3)
        if "Linear" in title:
            plt.axis([250, 1400, 0, 1])
        plt.legend(loc="best")

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    if save_plots == 1:
        plt.savefig('plt_' + title + '.png')
    else:
        plt.show()

def plot_all_boxplots_per_sample(df,parameter_to_plot,all_sample_IDs,versuchsplan=''):
    ##Change  'eta in %' to column index 4
    eta_index = parameter_to_plot.index('eta in %')
    parameter_to_plot.pop(eta_index)
    parameter_to_plot.insert(3, 'eta in %')
    sample_transposed = pd.DataFrame()
    plot_individual_plots = 0
    if plot_individual_plots ==1:
        for parameter in parameter_to_plot:
            for sample_i in all_sample_IDs:
                sample_transposed[sample_i]= (df.loc[df['Sample'] == sample_i, parameter])
                #sample_transposed.fillna(0, inplace=True)   #replaces Nan to zeros
                sample_transposed.fillna(sample_transposed[sample_i].median(), inplace=True)  #replaces Nan to the median

            plot_data = pd.DataFrame(sample_transposed, columns=all_sample_IDs).astype(float)
            plt.title(versuchsplan)
            plt.boxplot(plot_data, notch=False, patch_artist=True)
            ax = plt.gca()
            ax.set_ylabel(parameter)
            ax.set_xticklabels(all_sample_IDs)
            plt.show()
            plt.close()
            #plt.save()

    sample_transposed2 = pd.DataFrame()
    arrayb = []

    fig, axs = plt.subplots(2, 3, figsize=(8, 10))
    for i in range(len(all_sample_IDs)):
        arrayb.append(i + 1)

    for i, parameter in enumerate(parameter_to_plot):
        for sample_i in all_sample_IDs:
            sample_transposed2[sample_i] = df.loc[df['Sample'] == sample_i, parameter]
            #sample_transposed.fillna(0, inplace=True)   #replaces Nan to zeros
            sample_transposed2.fillna(sample_transposed2[sample_i].median(), inplace=True)  #replaces Nan to the median
        plot_data = pd.DataFrame(sample_transposed2, columns=all_sample_IDs).astype(float)
        axs[i // 3, i % 3].boxplot(plot_data.iloc[:, 0:len(all_sample_IDs)].values)
        axs[i // 3, i % 3].set_title(versuchsplan)
        axs[i // 3, i % 3].set_ylabel(parameter)
        axs[i // 3, i % 3].set_xlim([0,len(all_sample_IDs)+0.5])
        axs[i // 3, i % 3].set_xticks(arrayb)
        axs[i // 3, i % 3].set_xticklabels(list(all_sample_IDs), rotation=45, ha="right")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    plt.savefig('plt_summary_' + versuchsplan + '.png')

