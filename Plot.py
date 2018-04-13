import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class Plotter:

    def get_x_y(self, data, x_var, y_var):
        """
        This function can be used to get the input for a plot function
        based on the wanted x and y variables.
        :param data:
        :param x_var:
        :param y_var:
        :return:
        """
        x,y = [],[]
        for row in data:
            try:
                x.append(row[x_var])
                y.append(row[y_var])
            except KeyError:
                print("[ERROR] Invallid")
        return(x,y)

    def single_line(self,x, y):
        """
        This function can be used to make a one line plot
        :param x:  The x-coordinates
        :param y:  The y-coordinates
        """
        ax = plt.axes()
        ax.plot(x, y)
        plt.show()

    def double_line(self, x_list, y_list, legend_titles):
        """
        This function can be used to plot two lines in one figure
        :param x_list:  A nested list with x-coordinates
        :param y_list: A nested list with y-coordinates
        :param legend_titles: The titles that should be used in the legend
        """
        for x,y in zip(x_list, y_list):
            plt.plot(x, y)
        legend = ['y = {}'.format(legend_title) for legend_title in legend_titles]
        plt.legend(legend, loc='upper left')

        plt.show()

